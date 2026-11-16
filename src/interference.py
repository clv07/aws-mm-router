#!/usr/bin/env python3
"""Interference profiling: how much two workloads slow each other down when they share a node.

  python src/interference.py run --node c7a.2xlarge                 # on the node, like benchmark.py
  python src/interference.py run --runs 00 02 04 07 --pairs 02+04 04+07
  python src/interference.py summarize results/interference/<node>__<model>

`run` first measures each workload alone, then each pair on a llama-server with 2 slots. A pair
is two closed-loop streams, one per slot: stream A sends workload A again and again (a new input
each time, one request in flight), stream B the same with workload B, until both have finished
--repeats requests. The shorter workload therefore runs many times alongside one long request.
A random pause of up to --jitter s before each request shifts the two streams against each other,
so a request meets the other workload in every phase (prefill, image/audio encoding, decoding).

Every request is logged with its start, first-token and end time on a clock shared by the two
streams, and the gaps between its output tokens. `summarize` keeps the requests that ran almost
entirely (--min-overlap) alongside the other workload, compares them with the same workload run
alone, and writes the slowdown matrix: slowdown(A | B) = A's latency while B runs / A's latency
alone. A router can use it as an input: a request of A sent to a node whose other slot is busy
with B takes about slowdown(A | B) x A's solo time (load_matrix).

Run it on the node (in tmux), next to llama-server, like benchmark.py: the client then adds no
network time, but its own CPU use is kept small by building every payload before measuring.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from itertools import combinations_with_replacement
import json
from pathlib import Path
import random
import threading
import time

import requests

from benchmark import (OUTPUT_MODES, OUTPUT_TOKENS, RUNS, WARMUPS, RemoteSamples, append_rows, derived,
                       instance_type, median, payload, request_once, server_info)

OUT_DIR = Path("results/interference")   # one folder per node + model, as in results/phase1
DEFAULT_RUNS = ["00", "02", "04", "07"]   # one size per modality: chat, 1,024-token text, 1 image, 10 s audio
REPEATS = 5     # finished requests per stream (the shorter workload of a pair runs more often)
POOL = 40       # distinct inputs prebuilt per workload; reused in turn if a stream needs more
STALL_FACTOR = 5   # a gap between tokens longer than this x the request's median gap is a stall
REQUEST_FIELDS = ["run_id", "timestamp_utc", "model", "node", "slots", "output_mode", "pair",
                  "stream", "run", "modality", "size", "other_run", "index", "input",
                  "request_bytes", "start_s", "first_s", "end_s", "status", "error",
                  "finish_reason", "e2e_s", "ttft_s", "prompt_tokens", "output_tokens",
                  "prefill_tok_s", "decode_tok_s", "tpot_ms", "max_gap_s", "stall_s",
                  "server_prompt_ms", "server_predicted_ms"]
PAIR_FIELDS = ["run_id", "timestamp_utc", "model", "node", "slots", "output_mode", "pair",
               "run_a", "run_b", "wall_s", "requests_a", "requests_b", "failed", "peak_ram_gb",
               "cpu_pct"]
MATRIX_FIELDS = ["node", "model", "output_mode", "run", "other_run", "requests", "solo_e2e_s",
                 "e2e_s", "slowdown", "ttft_slowdown", "prefill_slowdown", "decode_slowdown",
                 "solo_max_gap_s", "max_gap_s", "stall_s", "pair_gain", "peak_ram_gb"]
REQUESTS_CSV, PAIRS_CSV, MATRIX_CSV = "requests.csv", "pairs.csv", "matrix.csv"


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def parse_pair(text, runs):
    a, sep, b = text.partition('+')
    if not sep or a not in RUNS or b not in RUNS:
        raise SystemExit(f'bad --pairs entry {text!r}: use <run>+<run>, e.g. 02+04 (runs 00-08)')
    missing = {a, b} - set(runs)
    if missing:
        raise SystemExit(f'--pairs {text}: add {" ".join(sorted(missing))} to --runs '
                         '(each workload in a pair is also measured alone)')
    return tuple(sorted((a, b)))


def pair_label(runs):
    return '+'.join(runs)


# ---------------------------------------------------------------- measuring

def build_pools(runs, size, output_mode):
    """`size` distinct payloads per workload (inputs after the warm-up ones), built before any
    measurement: tokenizing text and encoding images and audio would otherwise take CPU from
    llama-server while a pair runs."""
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


def token_gaps(times):
    """Largest gap between consecutive output tokens, and the time lost to stalls: the excess
    of every gap longer than STALL_FACTOR x the median gap (e.g. while the other slot's image
    or audio is being encoded)."""
    gaps = [b - a for a, b in zip(times, times[1:])]
    if not gaps:
        return None, None
    typical = median(gaps)
    return max(gaps), sum(g - typical for g in gaps if g > STALL_FACTOR * typical)


def measure(args, runs, pools, labels, requests_path, write_lock, seed):
    """Run one workload alone (runs = (A,)) or a pair (runs = (A, B)); return the request rows."""
    streams = list(runs)
    self_pair = len(streams) == 2 and streams[0] == streams[1]
    label = pair_label(runs)
    done = [0] * len(streams)
    lock = threading.Lock()
    wrapped = set()
    rows = []
    start = time.perf_counter()

    def loop(i):
        run = streams[i]
        other = streams[1 - i] if len(streams) == 2 else ''
        modality, setting = RUNS[run]
        items = pools[run][1:]
        rng = random.Random(f'{seed}:{label}:{i}')
        k = 0
        with requests.Session() as session:
            while True:
                with lock:
                    if all(d >= args.repeats for d in done) or done[i] >= args.max_requests:
                        return
                if len(streams) == 2 and args.jitter:
                    time.sleep(rng.uniform(0, args.jitter))
                # A self-pair's two streams take alternate inputs, so they never send the same one.
                j = 2 * k + i if self_pair else k
                if j >= len(items) and run not in wrapped:
                    wrapped.add(run)
                    print(f'  note: {run} needed more than {len(items)} inputs; reusing them '
                          '(raise --pool to avoid)', flush=True)
                index, body = items[j % len(items)]
                times = []
                sent = time.perf_counter() - start
                record = request_once(session, args.url, body, on_token=times.append)
                end = time.perf_counter() - start
                max_gap, stall = token_gaps(times)
                row = {**labels, 'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                       'pair': label, 'stream': 'AB'[i], 'run': run, 'modality': modality,
                       'size': setting, 'other_run': other, 'index': k + 1, 'input': index,
                       'request_bytes': len(json.dumps(body)), 'start_s': sent,
                       'first_s': sent + record['ttft_s'] if record['ttft_s'] is not None else None,
                       'end_s': end, **record, **derived(record),
                       'max_gap_s': max_gap, 'stall_s': stall}
                row.pop('queue_s')   # not meaningful here: each stream has its own slot
                with write_lock:
                    append_rows(requests_path, REQUEST_FIELDS, [row])
                    rows.append(row)
                print(f"  {label} {'AB'[i]} {run} #{k+1}: {record['status']} "
                      + (record['error'] or f"{record['e2e_s']:.2f}s"), flush=True)
                with lock:
                    done[i] += 1
                k += 1

    threads = [threading.Thread(target=loop, args=(i,)) for i in range(len(streams))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return rows, time.perf_counter() - start


def run(args):
    runs = sorted(set(args.runs))
    pairs = ([parse_pair(p, runs) for p in args.pairs] if args.pairs
             else [p for p in combinations_with_replacement(runs, 2)
                   if not (args.no_self and p[0] == p[1])])
    node = args.node or instance_type()
    model, build, slots, ctx = server_info(args.url)
    out_dir = OUT_DIR / f'{node}__{model}'
    if args.where:
        print(out_dir)
        return
    if slots is not None and slots < 2:
        raise SystemExit(f'llama-server has {slots} slot; a pair needs 2. Restart it with '
                         'SLOTS=2 ./src/run.sh <model> (on an 8 GB node check the memory first)')
    run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    print(f'run_id={run_id} node={node} model={model} build={build} slots={slots} ctx_per_slot={ctx} '
          f'output={args.output} alone: {" ".join(runs)} pairs: {" ".join(map(pair_label, pairs))} '
          f'-> {out_dir}', flush=True)
    pools = build_pools(runs, args.pool, args.output)
    labels = {'run_id': run_id, 'model': model, 'node': node, 'slots': slots,
              'output_mode': args.output}
    write_lock = threading.Lock()

    with requests.Session() as session:
        for r in runs:
            record = request_once(session, args.url, pools[r][0][1])
            print(f"  warmup {r}: {record['status']} " + (record['error'] or f"{record['e2e_s']:.2f}s"),
                  flush=True)

    for runs_in_play in [(r,) for r in runs] + pairs:
        label = pair_label(runs_in_play)
        print(f'[{node}] {label}' + (' alone' if len(runs_in_play) == 1 else ' together'), flush=True)
        with RemoteSamples(args.ssh) as monitor:
            rows, wall = measure(args, runs_in_play, pools, labels, out_dir / REQUESTS_CSV,
                                 write_lock, args.seed)
        if monitor.problem:
            print('  CPU/RAM monitoring:', monitor.problem)
        cpu, ram = monitor.metrics()
        count = lambda s: sum(1 for r in rows if r['stream'] == s)
        append_rows(out_dir / PAIRS_CSV, PAIR_FIELDS, [{
            **labels, 'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'pair': label, 'run_a': runs_in_play[0],
            'run_b': runs_in_play[1] if len(runs_in_play) == 2 else '', 'wall_s': wall,
            'requests_a': count('A'), 'requests_b': count('B'),
            'failed': sum(1 for r in rows if r['status'] != 'ok'),
            'peak_ram_gb': ram, 'cpu_pct': cpu}])
    print(f'Appended results to {out_dir}/{REQUESTS_CSV} and {PAIRS_CSV} (run_id={run_id}); '
          f'next: python src/interference.py summarize {out_dir}')


# ---------------------------------------------------------------- summarizing

def overlap(row, others):
    """Fraction of this request's lifetime during which the other stream had a request in flight."""
    s, e = float(row['start_s']), float(row['end_s'])
    if e <= s:
        return 0.0
    covered = sum(max(0.0, min(e, float(o['end_s'])) - max(s, float(o['start_s']))) for o in others)
    return covered / (e - s)


def fnum(row, field):
    return float(row[field]) if row.get(field) not in (None, '') else None


def medians(rows, field):
    return median([v for v in (fnum(r, field) for r in rows) if v is not None])


def largest(rows, field):
    return max((v for v in (fnum(r, field) for r in rows) if v is not None), default=None)


def mean(rows, field):
    values = [v for v in (fnum(r, field) for r in rows) if v is not None]
    return sum(values) / len(values) if values else None


def ratio(a, b):
    return a / b if a is not None and b else None


def summarize_rows(rows, pair_rows, min_overlap):
    """Matrix rows: one per (output mode, workload, what ran alongside), 'solo' for alone."""
    ok = [r for r in rows if r['status'] == 'ok']
    by_pair = defaultdict(list)   # (run_id, pair) -> rows, to find each request's partner stream
    for r in ok:
        by_pair[(r['run_id'], r['pair'])].append(r)
    ram = defaultdict(list)
    for p in pair_rows:
        if p['peak_ram_gb']:
            ram[(p['output_mode'], p['pair'])].append(float(p['peak_ram_gb']))

    groups = defaultdict(list)    # (output mode, run, other run or '') -> requests
    for (_, pair), members in by_pair.items():
        for r in members:
            if not r['other_run']:
                groups[(r['output_mode'], r['run'], '')].append(r)
                continue
            partner = [o for o in members if o['stream'] != r['stream']]
            if overlap(r, partner) >= min_overlap:
                groups[(r['output_mode'], r['run'], r['other_run'])].append(r)

    first = ok[0] if ok else {}
    out = {}
    for (mode, run, other), members in sorted(groups.items()):
        solo = groups.get((mode, run, ''), [])
        e2e, solo_e2e = medians(members, 'e2e_s'), medians(solo, 'e2e_s')
        out[(mode, run, other)] = {
            'node': first.get('node'), 'model': first.get('model'), 'output_mode': mode,
            'run': run, 'other_run': other or 'solo', 'requests': len(members),
            'solo_e2e_s': solo_e2e, 'e2e_s': e2e,
            'slowdown': ratio(e2e, solo_e2e),
            'ttft_slowdown': ratio(medians(members, 'ttft_s'), medians(solo, 'ttft_s')),
            'prefill_slowdown': ratio(medians(solo, 'prefill_tok_s'), medians(members, 'prefill_tok_s')),
            'decode_slowdown': ratio(medians(solo, 'decode_tok_s'), medians(members, 'decode_tok_s')),
            # Stalls hit only the requests decoding when the other one starts its prefill or
            # encoding, so the worst gap and the mean stall rather than medians.
            'solo_max_gap_s': largest(solo, 'max_gap_s'), 'max_gap_s': largest(members, 'max_gap_s'),
            'stall_s': mean(members, 'stall_s'),
            'peak_ram_gb': max(ram.get((mode, pair_label(sorted((run, other)) if other else [run])), [None]),
                               key=lambda v: v or 0)}
    # Combined throughput of the pair relative to running its two workloads one after the other
    # (weighted speedup): 1/slowdown(A|B) + 1/slowdown(B|A). 1 = sharing the node gains nothing,
    # 2 = neither slows the other; below 1 = they would be faster one at a time.
    for (mode, run, other), m in out.items():
        back = out.get((mode, other, run)) if other else None
        if other and m['slowdown'] and back and back['slowdown']:
            m['pair_gain'] = 1 / m['slowdown'] + 1 / back['slowdown']
    return list(out.values())


def print_matrix(matrix, field, title):
    modes = sorted({m['output_mode'] for m in matrix})
    for mode in modes:
        cells = {(m['run'], m['other_run']): m.get(field) for m in matrix if m['output_mode'] == mode}
        runs = sorted({r for r, _ in cells})
        others = sorted({o for _, o in cells if o != 'solo'})
        name = lambda r: f'{r} {RUNS[r][0]}{"" if RUNS[r][0] == "chat" else RUNS[r][1]}'
        print(f'\n{title} (output={mode}); rows: the request measured, columns: what ran alongside')
        print(f'  {"":<12}' + ''.join(f'{name(o):>12}' for o in others))
        for r in runs:
            cell = lambda v: (f'{"-":>12}' if v is None else f'{v:>12}' if isinstance(v, int)
                              else f'{v:>12.2f}')
            print(f'  {name(r):<12}' + ''.join(cell(cells.get((r, o))) for o in others))


def summarize(args):
    folder = Path(args.folder)
    rows = read_csv(folder / REQUESTS_CSV)
    pair_rows = read_csv(folder / PAIRS_CSV) if (folder / PAIRS_CSV).exists() else []
    if args.run_id:
        rows = [r for r in rows if r['run_id'] in args.run_id]
        pair_rows = [p for p in pair_rows if p['run_id'] in args.run_id]
    matrix = summarize_rows(rows, pair_rows, args.min_overlap)
    with (folder / MATRIX_CSV).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=MATRIX_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(matrix)
    print_matrix(matrix, 'slowdown', 'Slowdown: latency alongside / latency alone')
    print_matrix(matrix, 'pair_gain', 'Pair gain: 1/slowdown(A|B) + 1/slowdown(B|A); 1 = no gain from sharing')
    print_matrix(matrix, 'requests', f'Requests used (overlap >= {args.min_overlap})')
    print(f'\nwrote {folder / MATRIX_CSV}')


def load_matrix(folder, output_mode='fixed'):
    """For a router: {(run, other_run): slowdown} from a summarized folder, plus the same keyed
    by modality (median over that modality's measured sizes) for workloads not profiled, e.g.
    look up ('03', '04') and fall back to ('text', 'image'). Solo entries are left out."""
    rows = [r for r in read_csv(Path(folder) / MATRIX_CSV)
            if r['output_mode'] == output_mode and r['other_run'] != 'solo' and r['slowdown']]
    result = {(r['run'], r['other_run']): float(r['slowdown']) for r in rows}
    by_modality = defaultdict(list)
    for (a, b), v in list(result.items()):
        by_modality[(RUNS[a][0], RUNS[b][0])].append(v)
    result.update({k: median(v) for k, v in by_modality.items()})
    return result


def main():
    parser = argparse.ArgumentParser(description='Pairwise interference profiling on one node.')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run', help='measure each workload alone and each pair together (on the node)')
    p.add_argument('--runs', nargs='+', choices=RUNS.keys(), default=DEFAULT_RUNS,
                   help='workloads (benchmark.py RUNS) to measure alone and in pairs; default: '
                        + ' '.join(DEFAULT_RUNS))
    p.add_argument('--pairs', nargs='+', help='only these pairs, e.g. 02+04 04+07 '
                                              '(default: every pair of --runs, including each with itself)')
    p.add_argument('--no-self', action='store_true',
                   help='leave out a workload paired with itself (benchmark.py --concurrency 2 measures it too)')
    p.add_argument('--repeats', type=int, default=REPEATS, help='finished requests per stream')
    p.add_argument('--max-requests', type=int, default=200, help='safety cap per stream')
    p.add_argument('--pool', type=int, default=POOL, help='distinct inputs prebuilt per workload')
    p.add_argument('--jitter', type=float, default=0.5,
                   help='random pause of up to this many seconds before each request of a pair')
    p.add_argument('--seed', type=int, default=1, help='seed for the pauses')
    p.add_argument('--output', choices=OUTPUT_MODES, default='fixed',
                   help=f'fixed (default): every request generates {OUTPUT_TOKENS} tokens, so latency does not depend '
                        'on the answer; natural: stop at end of answer')
    p.add_argument('--run-id', help='label for all rows of this invocation; default: current UTC time')
    p.add_argument('--where', action='store_true', help='print the results folder and exit')
    p.add_argument('--node', help='node label; default: this EC2 instance type')
    p.add_argument('--url', default='http://127.0.0.1:8080', help='llama-server address')
    p.add_argument('--ssh', help='sample CPU/RAM on this SSH host instead of locally')
    p.set_defaults(func=run)
    s = sub.add_parser('summarize', help='slowdown matrix from a results folder')
    s.add_argument('folder', help='results/interference/<node>__<model>')
    s.add_argument('--min-overlap', type=float, default=0.8,
                   help='use a paired request only if the other stream was busy for at least this '
                        'fraction of its lifetime')
    s.add_argument('--run-id', nargs='+', help='only these run ids (default: all)')
    s.set_defaults(func=summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
