#!/usr/bin/env python3
"""Co-running profiler: groups of closed-loop request streams share one llama-server, to measure how much
requests slow each other down (pairwise interference) and how cost grows with k requests in flight.

  python src/corun.py pairs --node c7a.2xlarge-p2 --url http://10.0.1.12:8080      # alone, then every pair
  python src/corun.py pairs --node c7a.2xlarge-p2 --url http://10.0.1.12:8080 --runs 00 02 04 07 --pairs 02+04 04+07
  python src/corun.py groups --node c7a.2xlarge-p2 --url http://127.0.0.1:18080 --ssh aws-c7a \
         --mix default --allow-queue --groups 00+mix 00+mix+mix mix+mix mix+mix+mix+mix

Runs on the router machine (the client), which sends requests to the serving node over the network. CPU/RAM is
sampled on the serving node over SSH (default: the --url host). Every row records the router, so the network part
of each request's overhead can be fitted per (node, router).
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations_with_replacement
import json
import math
from pathlib import Path
import random
import threading
import time
from urllib.parse import urlparse

import requests

from benchmark import (OUTPUT_MODES, OUTPUT_TOKENS, RUNS, WARMUPS, RemoteSamples, append_rows, derived,
                       instance_type, payload, request_once, server_info)
from tracegen import MIXES, parse_mix

MIX = 'mix'           # stream spec that draws each request's workload from --mix
DEFAULT_RUNS = ["00", "02", "04", "07"]   # one size per modality: chat, 1,024-token text, 1 image, 10 s audio
MAX_WALL_S = 2400     # a group stops sending new requests after this long (groups only)
MAX_REQUESTS = 200    # safety cap per stream
OVERLAP_S = 0.5
REQUESTS_CSV = 'requests.csv'

PAIR_REQUEST_FIELDS = ["run_id", "timestamp_utc", "model", "node", "router", "slots", "output_mode", "pair",
                       "stream", "run", "modality", "size", "other_run", "index", "input",
                       "request_bytes", "start_s", "first_s", "end_s", "status", "error",
                       "finish_reason", "e2e_s", "ttft_s", "prompt_tokens", "output_tokens",
                       "prefill_tok_s", "decode_tok_s", "tpot_ms", "token_s",
                       "server_prompt_ms", "server_predicted_ms"]
PAIR_FIELDS = ["run_id", "timestamp_utc", "model", "node", "router", "slots", "output_mode", "pair",
               "run_a", "run_b", "wall_s", "requests_a", "requests_b", "failed", "peak_ram_gb",
               "cpu_pct"]
GROUP_REQUEST_FIELDS = ["run_id", "timestamp_utc", "model", "node", "router", "slots", "output_mode", "group", "k",
                        "stream", "spec", "run", "modality", "size", "index", "input", "request_bytes",
                        "inflight_at_send", "others_at_send", "start_s", "svc_start_s", "first_s", "end_s",
                        "status", "error", "finish_reason", "e2e_s", "service_s", "overhead_s", "ttft_s",
                        "prompt_tokens", "output_tokens", "prefill_tok_s", "decode_tok_s", "tpot_ms",
                        "token_s", "server_prompt_ms", "server_predicted_ms"]
GROUP_FIELDS = ["run_id", "timestamp_utc", "model", "node", "router", "slots", "output_mode", "group", "k",
                "mix", "wall_s", "requests", "failed", "req_s", "peak_ram_gb", "cpu_pct"]


@dataclass(frozen=True)
class Experiment:
    out_dir: Path
    summary_csv: str
    request_fields: list[str]
    summary_fields: list[str]
    pairwise: bool


EXPERIMENTS = {
    'pairs': Experiment(Path('results/interference'), 'pairs.csv', PAIR_REQUEST_FIELDS, PAIR_FIELDS, pairwise=True),
    'groups': Experiment(Path('results/concurrency'), 'groups.csv', GROUP_REQUEST_FIELDS, GROUP_FIELDS, pairwise=False),
}


def utc_now() -> str:
    """Current UTC time as an ISO timestamp"""
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def append_row(path: Path, fields: list[str], row: dict) -> None:
    """Append one row to a results CSV"""
    append_rows(path, fields, [{f: row.get(f) for f in fields}])


# groups
def pair_groups(args: argparse.Namespace) -> list[list[str]]:
    """build pair of runs"""
    runs = sorted(set(args.runs))
    if not args.pairs:
        pairs = [list(p) for p in combinations_with_replacement(runs, 2)
                 if not (args.no_self and p[0] == p[1])]
        return [[r] for r in runs] + pairs
    pairs = []
    for text in args.pairs:
        a, sep, b = text.partition('+')
        if not sep or a not in RUNS or b not in RUNS:
            raise SystemExit(f'bad --pairs entry {text!r}: use <run>+<run>, e.g. 02+04 (runs 00-08)')
        missing = {a, b} - set(runs)
        if missing:
            raise SystemExit(f'--pairs {text}: add {" ".join(sorted(missing))} to --runs '
                             '(each workload in a pair is also measured alone)')
        pairs.append(sorted((a, b)))
    return [[r] for r in runs] + pairs


def parse_groups(args: argparse.Namespace, slots: int | None) -> list[list[str]]:
    """Create group of runs. Refuses bad, duplicate entries, and # streams > # slots unless --allow-queue"""
    groups = [text.split('+') for text in args.groups]
    for text, specs in zip(args.groups, groups):
        if not all(s in RUNS or s == MIX for s in specs):
            raise SystemExit(f'bad --groups entry {text!r}: join run ids (00-08) or "{MIX}" with +, e.g. 00+mix+mix')
    labels = ['+'.join(g) for g in groups]
    if len(set(labels)) != len(labels):
        raise SystemExit('--groups lists a group twice; its rows would share run_id and group')
    if slots is None:
        print('warning: llama-server did not report its slot count; cannot check k against slots', flush=True)
        return groups
    too_big = [label for label, g in zip(labels, groups) if len(g) > slots]
    if too_big and not args.allow_queue:
        raise SystemExit(f'{" ".join(too_big)}: more streams than the server\'s {slots} slots; pass --allow-queue '
                         'to measure waiting in llama-server\'s queue')
    return groups


def monitor_target(args: argparse.Namespace) -> str | None:
    """SSH host to sample the serving node's CPU/RAM on: --ssh, else the --url host; None samples locally."""
    if args.ssh == 'local':
        return None
    host = args.ssh or urlparse(args.url).hostname
    if host in ('127.0.0.1', 'localhost', '::1'):
        raise SystemExit(f'--url {args.url} is on this machine (a tunnel?): pass --ssh <serving node>, '
                         'or --ssh local if llama-server runs here')
    return host


def runs_needed(groups: list[list[str]], mix: dict[str, float]) -> list[str]:
    """Every workload a group can send: its run ids, plus every run in --mix if a stream is 'mix'."""
    runs = {s for g in groups for s in g if s != MIX}
    if any(MIX in g for g in groups):
        runs |= {r for r, w in mix.items() if w > 0}
    return sorted(runs)


# measuring

def build_pools(runs: list[str], size: int, output_mode: str) -> dict[str, list[tuple[int, dict]]]:
    """Build data pools."""
    from loaders.coco import COCODataset
    from loaders.cnn_dailymail import CNNDailyMailDataset
    from loaders.dolly import DollyQADataset
    from loaders.librispeech import LibriSpeechDataset
    classes = {'chat': DollyQADataset, 'text': CNNDailyMailDataset,
               'image': COCODataset, 'audio': LibriSpeechDataset}
    datasets, pools = {}, {}
    for run in runs:
        modality, setting = RUNS[run]
        ds = datasets.setdefault(modality, classes[modality]())
        print(f'  building {size} inputs for {run} {modality}={setting}', flush=True)
        pools[run] = [(WARMUPS + k, payload(ds, modality, setting, WARMUPS + k, output_mode))
                      for k in range(size)]
        pools[run].insert(0, (0, payload(ds, modality, setting, 0, output_mode)))   # warm-up
    return pools


def warm_up(url: str, pools: dict) -> None:
    """One request per workload (pool item 0) before measuring."""
    with requests.Session() as session:
        for run, items in pools.items():
            record = request_once(session, url, items[0][1])
            print(f"  warmup {run}: {record['status']} " + (record['error'] or f"{record['e2e_s']:.2f}s"),
                  flush=True)


def service_split(record: dict) -> tuple[float, float] | None:
    """(service_s, overhead_s) of a successful request from llama-server's timings, else None.

    service_s = (server_prompt_ms + server_predicted_ms)/1000 and overhead_s = e2e_s - service_s
    (network, HTTP and request parsing, plus waiting in llama-server's queue)."""
    prompt, predicted, e2e = (record.get(f) for f in ('server_prompt_ms', 'server_predicted_ms', 'e2e_s'))
    if record.get('status') != 'ok' or prompt is None or predicted is None or e2e is None:
        return None
    service = (prompt + predicted) / 1000
    return service, e2e - service


def measure(args: argparse.Namespace, exp: Experiment, specs: list[str], pools: dict, mix: dict[str, float],
            labels: dict, requests_path: Path, write_lock: threading.Lock) -> tuple[list[dict], float]:
    """Run one group of closed-loop streams until each has finished --repeats requests (or --max-wall
    passes); return the request rows and the wall time."""
    k = len(specs)
    group = '+'.join(specs)
    self_pair = k == 2 and specs[0] == specs[1]
    done = [0] * k
    inflight = {}   # stream -> run of its request in flight
    lock = threading.Lock()
    wrapped = set()
    rows = []
    start = time.perf_counter()

    def finished(i):
        return (all(d >= args.repeats for d in done) or done[i] >= args.max_requests
                or time.perf_counter() - start >= args.max_wall)

    def loop(i):
        spec = specs[i]
        stream = 'AB'[i] if exp.pairwise else i
        # Separate seeded streams, so the workload sequence does not depend on the pauses.
        pauses = random.Random(f'{args.seed}:{group}:{i}' + ('' if exp.pairwise else ':jitter'))
        draws = random.Random(f'{args.seed}:{group}:{i}:mix')
        c = 0
        with requests.Session() as session:
            while True:
                with lock:
                    if finished(i):
                        return
                if k >= 2 and args.overlap:
                    time.sleep(pauses.uniform(0, args.overlap))
                run = draws.choices(list(mix), weights=list(mix.values()))[0] if spec == MIX else spec
                modality, setting = RUNS[run]
                items = pools[run][1:]
                # Streams never send the same input; in pairs only a self-pair can collide.
                if exp.pairwise:
                    j = 2 * c + i if self_pair else c
                else:
                    j = c * k + i
                times = []
                with lock:
                    if j >= len(items) and run not in wrapped:
                        wrapped.add(run)
                        print(f'  note: {run} needed more than {len(items)} inputs; reusing them '
                              '(raise --pool to avoid)', flush=True)
                    others = sorted(inflight.values())
                    inflight[i] = run
                    sent = time.perf_counter() - start
                index, body = items[j % len(items)]
                record = request_once(session, args.url, body, on_token=times.append)
                end = time.perf_counter() - start
                with lock:
                    del inflight[i]
                service, overhead = service_split(record) or (None, None)
                # derived()'s queue_s is not written: the analysis derives it from overhead_s.
                row = {**labels, 'timestamp_utc': utc_now(), 'pair': group, 'group': group, 'k': k,
                       'stream': stream, 'spec': spec, 'run': run, 'modality': modality, 'size': setting,
                       'other_run': '+'.join(specs[:i] + specs[i + 1:]), 'index': c + 1, 'input': index,
                       'request_bytes': len(json.dumps(body)),
                       'inflight_at_send': len(others), 'others_at_send': '+'.join(others),
                       'start_s': sent, 'svc_start_s': end - service if service is not None else None,
                       'first_s': sent + record['ttft_s'] if record['ttft_s'] is not None else None,
                       'end_s': end, **record, **derived(record), 'service_s': service,
                       'overhead_s': overhead,
                       # arrival time of every output token, on the group's clock like start_s/end_s
                       'token_s': ' '.join(f'{x - start:.3f}' for x in times)}
                with write_lock:
                    append_row(requests_path, exp.request_fields, row)
                    rows.append(row)
                print(f"  {group} [{stream}] {run} #{c + 1}: {record['status']} "
                      + (record['error'] or f"{record['e2e_s']:.2f}s")
                      + (f" (with {'+'.join(others)})" if others else ''), flush=True)
                with lock:
                    done[i] += 1
                c += 1

    threads = [threading.Thread(target=loop, args=(i,)) for i in range(k)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return rows, time.perf_counter() - start


def summary_row(labels: dict, specs: list[str], mix_spec: str, rows: list[dict], wall: float,
                monitor: RemoteSamples) -> dict:
    """The pairs.csv / groups.csv row of one measured group."""
    cpu, ram = monitor.metrics()
    ok = sum(1 for r in rows if r['status'] == 'ok')
    count = lambda s: sum(1 for r in rows if r['stream'] == s)
    return {**labels, 'timestamp_utc': utc_now(), 'pair': '+'.join(specs), 'group': '+'.join(specs),
            'k': len(specs), 'run_a': specs[0], 'run_b': specs[1] if len(specs) == 2 else '',
            'requests_a': count('A'), 'requests_b': count('B'), 'mix': mix_spec if MIX in specs else '',
            'wall_s': wall, 'requests': len(rows), 'failed': len(rows) - ok,
            'req_s': ok / wall if wall else None, 'peak_ram_gb': ram, 'cpu_pct': cpu}


def run(args: argparse.Namespace) -> None:
    """Measure every group in turn and append its requests and summary row."""
    exp = EXPERIMENTS[args.command]
    mix_spec = MIXES.get(args.mix, args.mix) if args.mix else ''
    mix = parse_mix(mix_spec) if mix_spec else {}
    node, router = args.node, instance_type()   # router: this machine, the client (EC2 type or hostname)
    ssh = monitor_target(args)
    model, build, slots, ctx = server_info(args.url)
    out_dir = exp.out_dir / f'{node}__{model}'
    if args.where:
        print(out_dir)
        return
    groups = pair_groups(args) if exp.pairwise else parse_groups(args, slots)
    names = ['+'.join(g) for g in groups]
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')  # labels every row of this invocation
    if exp.pairwise and slots is not None and slots < 2:
        raise SystemExit(f'llama-server has {slots} slot; a pair needs 2. Restart it with '
                         'SLOTS=2 ./src/run.sh <model> (on an 8 GB node check the memory first)')
    print(f'run_id={run_id} node={node} router={router} cpu/ram={ssh or "local"} model={model} build={build} slots={slots} ctx_per_slot={ctx} '
          f'output={args.output}' + (f' mix={mix_spec}' if mix_spec else '')
          + f' groups: {" ".join(names)} -> {out_dir}', flush=True)
    pools = build_pools(runs_needed(groups, mix), args.pool, args.output)
    labels = {'run_id': run_id, 'model': model, 'node': node, 'router': router, 'slots': slots, 'output_mode': args.output}
    write_lock = threading.Lock()
    if not args.no_warmup:
        warm_up(args.url, pools)

    for specs in groups:
        print(f'[{node}] {"+".join(specs)} (k={len(specs)})', flush=True)
        with RemoteSamples(ssh) as monitor:
            rows, wall = measure(args, exp, specs, pools, mix, labels, out_dir / REQUESTS_CSV, write_lock)
        if monitor.problem:
            print('  CPU/RAM monitoring:', monitor.problem)
        per_stream = Counter(r['stream'] for r in rows)
        if len(per_stream) < len(specs) or min(per_stream.values()) < args.repeats:
            print(f'  note: a stream finished fewer than --repeats {args.repeats} requests (--max-wall/--max-requests)')
        append_row(out_dir / exp.summary_csv, exp.summary_fields,
                   summary_row(labels, specs, mix_spec, rows, wall, monitor))
    print(f'Appended results to {out_dir}/{REQUESTS_CSV} and {exp.summary_csv} (run_id={run_id})')


def add_common(p: argparse.ArgumentParser, repeats: int, pool: int) -> None:
    """Options shared by both subcommands; repeats and pool defaults differ."""
    p.add_argument('--repeats', type=int, default=repeats, help='finished requests per stream')
    p.add_argument('--max-requests', type=int, default=MAX_REQUESTS, help='safety cap per stream')
    p.add_argument('--pool', type=int, default=pool, help='distinct inputs prebuilt per workload')
    p.add_argument('--overlap', type=float, default=OVERLAP_S,
                   help='varies how requests overlap: random pause of up to this many seconds before each '
                        'request when k >= 2')
    p.add_argument('--seed', type=int, default=1, help='seed for the pauses and the mix draws')
    p.add_argument('--output', choices=OUTPUT_MODES, default='fixed',
                   help=f'fixed (default): every request generates {OUTPUT_TOKENS} tokens, so latency does not depend '
                        'on the answer; natural: stop at end of answer')
    p.add_argument('--where', action='store_true', help='print the results folder and exit')
    p.add_argument('--node', required=True,
                   help='label of the serving node, e.g. c7a.2xlarge-p2; names the results folder')
    p.add_argument('--url', default='http://127.0.0.1:8080', help="the serving node's llama-server address")
    p.add_argument('--ssh', help="SSH host of the serving node, to sample its CPU/RAM; default: the --url host "
                                 "(required when --url is a tunnel); 'local' samples this machine")


def main() -> None:
    parser = argparse.ArgumentParser(description='Co-running profiler: closed-loop streams sharing one llama-server.')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('pairs', help='interference: each workload alone, then each pair together (2 slots)')
    p.add_argument('--runs', nargs='+', choices=RUNS.keys(), default=DEFAULT_RUNS,
                   help='workloads (benchmark.py RUNS) to measure alone and in pairs; default: '
                        + ' '.join(DEFAULT_RUNS))
    p.add_argument('--pairs', nargs='+', help='only these pairs, e.g. 02+04 04+07 '
                                              '(default: every pair of --runs, including each with itself)')
    p.add_argument('--no-self', action='store_true',
                   help='leave out a workload paired with itself (benchmark.py --concurrency 2 measures it too)')
    add_common(p, repeats=5, pool=40)
    p.set_defaults(mix=None, allow_queue=False, max_wall=math.inf, no_warmup=False)

    g = sub.add_parser('groups', help='dynamic concurrency: groups of k streams, fixed workloads or drawn from --mix')
    g.add_argument('--groups', nargs='+', required=True,
                   help=f'groups of +-joined streams, each a run id (00-08) or "{MIX}" (drawn from --mix), '
                        'e.g. 00 00+mix mix+mix+mix; k = number of streams')
    g.add_argument('--mix', default='default', help=f'workload mix of "{MIX}" streams: one of {", ".join(MIXES)}, '
                                                    'or chat=..,text=.. / run ids')
    g.add_argument('--allow-queue', action='store_true',
                   help="allow more streams than slots: the extra requests wait in llama-server's queue")
    g.add_argument('--max-wall', type=float, default=MAX_WALL_S,
                   help='stop sending new requests of a group after this many seconds')
    g.add_argument('--no-warmup', action='store_true',
                   help='skip the warm-up request per workload (e.g. right after another run on the same server)')
    add_common(g, repeats=4, pool=30)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
