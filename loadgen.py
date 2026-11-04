#!/usr/bin/env python3
"""Phase 2 open-loop load generator and router over several llama-server nodes.

  python loadgen.py trace --rate 0.2 --duration 900 --seed 1     # writes traces/<name>.csv
  python loadgen.py capacity --nodes nodes.json                   # predicted cost per node, capacity
  python loadgen.py run --trace traces/<name>.csv --nodes nodes.json --policy ect --cluster C

A trace fixes every request's arrival time, workload (benchmark.py RUNS) and input, so every
policy and cluster replays exactly the same traffic. Each request is sent at its arrival time
whether or not earlier ones have finished (open loop). The policy picks the node on arrival;
the request then waits in that node's llama-server queue if all of its slots are busy.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import threading
import time

import requests

from benchmark import (RESULTS_DIR, RUNS, RemoteSamples, append_rows, derived, median, payload,
                       request_once, server_info)

TRACES_DIR = Path("traces")
OUT_DIR = Path("results/phase2")
DEFAULT_MIX = "chat=50,text=20,image=15,audio=15"   # percent per modality; sizes uniform within it
POLICIES = ("random", "rr", "least", "ect")
TRACE_FIELDS = ["id", "arrival_s", "run", "modality", "size", "input"]
REQUEST_FIELDS = ["run_id", "timestamp_utc", "cluster", "trace", "policy", "seed", "output_mode",
                  "id", "arrival_s", "sent_s", "node", "model", "node_inflight", "predicted_s",
                  "predicted_e2e_s", "run", "modality", "size", "input", "status", "error",
                  "finish_reason", "e2e_s", "ttft_s", "queue_s", "prompt_tokens", "output_tokens",
                  "prefill_tok_s", "decode_tok_s", "tpot_ms", "server_prompt_ms",
                  "server_predicted_ms"]
SUMMARY_FIELDS = ["run_id", "timestamp_utc", "cluster", "trace", "policy", "seed", "output_mode",
                  "offered_req_s", "group", "requests", "success", "failed", "median_e2e_s",
                  "p95_e2e_s", "p99_e2e_s", "median_ttft_s", "p95_ttft_s", "median_queue_s",
                  "req_s", "usd_per_1k_requests", "peak_ram_gb", "cpu_pct"]


def percentile(values, q):
    xs = sorted(values)
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- traces

def parse_mix(text):
    """'chat=50,text=20,...' -> probability per run id. A key is a modality (its sizes share the
    weight equally) or a run id from benchmark.py RUNS, e.g. '03=10' for 3072-token text only."""
    weights = {}
    for part in text.split(','):
        key, _, weight = part.partition('=')
        key = key.strip()
        runs = [key] if key in RUNS else [r for r, (m, _) in RUNS.items() if m == key]
        if not runs or not weight:
            raise SystemExit(f'bad --mix entry {part!r}: use <modality>=<weight> '
                             f'(chat, text, image, audio) or <run id>=<weight> (00-08)')
        for r in runs:
            weights[r] = weights.get(r, 0) + float(weight) / len(runs)
    total = sum(weights.values())
    return {r: w / total for r, w in sorted(weights.items())}


def make_trace(args):
    mix = parse_mix(args.mix)
    rng = random.Random(args.seed)
    rows, t, used = [], 0.0, {}
    while True:
        # Poisson arrivals. Two draws per request, so the same seed at another --rate gives the
        # same requests in the same order, with every gap scaled by the rate ratio.
        t += rng.expovariate(args.rate)
        run = rng.choices(list(mix), weights=list(mix.values()))[0]
        if t > args.duration:
            break
        modality, size = RUNS[run]
        used[modality] = used.get(modality, 0) + 1   # a new input per request; input 0 = warm-up
        rows.append({'id': len(rows) + 1, 'arrival_s': f'{t:.3f}', 'run': run,
                     'modality': modality, 'size': size, 'input': used[modality]})
    out = Path(args.out or TRACES_DIR / f'rate{args.rate:g}-dur{args.duration:g}-seed{args.seed}.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = {m: sum(r['modality'] == m for r in rows) for m in ('chat', 'text', 'image', 'audio')}
    print(f'wrote {out}: {len(rows)} requests over {args.duration:g} s '
          f'({len(rows) / args.duration:.3f} req/s), {counts}')


# ---------------------------------------------------------------- node costs from Phase 1

def load_nodes(path):
    """nodes.json: {"<name>": {"url": ..., "profile": "<results/phase1 folder>", "ssh": optional,
    "price_per_hour": optional, "slot_gain": optional}, ...}; entry order = round-robin order."""
    nodes = json.loads(Path(path).read_text())
    for name, cfg in nodes.items():
        for key in ('url', 'profile'):
            if key not in cfg:
                raise SystemExit(f'{path}: node {name!r} has no "{key}"')
        if not (RESULTS_DIR / cfg['profile']).is_dir():
            raise SystemExit(f'{path}: node {name!r}: no folder {RESULTS_DIR / cfg["profile"]}')
    return nodes


def profile_costs(profile):
    """Median e2e_s per workload at concurrency 1, natural output, in one Phase 1 results folder
    (summary_v2.csv rows replace summary.csv rows; later rows replace earlier ones); workloads
    with any failed request are left out."""
    costs = {}
    for file in ('summary.csv', 'summary_v2.csv'):
        path = RESULTS_DIR / profile / file
        for r in read_csv(path) if path.exists() else []:
            if (r.get('concurrency') == '1' and r.get('output_mode', 'natural') == 'natural'
                    and r['failed'] == '0' and r['median_e2e_s']):
                costs[r['run']] = float(r['median_e2e_s'])
    return costs


def load_costs(nodes):
    """Predicted service time (s) of each workload on each node when it runs alone, from the
    node's Phase 1 results. A workload not measured on a node (e.g. failed audio) is estimated
    from every other Phase 1 folder that has it: that folder's time times this node's median
    ratio to it on the workloads both have; the median of those estimates is used."""
    others = {d.name: profile_costs(d.name) for d in sorted(RESULTS_DIR.iterdir()) if d.is_dir()}
    costs, estimated = {}, []
    for name, cfg in nodes.items():
        own = dict(others[cfg['profile']])
        if not own:
            raise SystemExit(f'node {name!r}: no concurrency-1 results in {cfg["profile"]}')
        for run in RUNS:
            if run in own:
                continue
            guesses = [ref[run] * median([own[r] / ref[r] for r in own if r in ref])
                       for p, ref in others.items()
                       if p != cfg['profile'] and run in ref and set(own) & set(ref)]
            if not guesses:
                raise SystemExit(f'no Phase 1 folder has a concurrency-1 result for run {run}')
            own[run] = median(guesses)
            estimated.append(f'{name}:{run}')
        costs[name] = own
    return costs, estimated


def slot_gain(profile):
    """How much faster a node gets through its work with 2 requests at once than with 1: median
    over workloads of req_s at concurrency 2 / concurrency 1 (2 slots, fixed output) in the
    Phase 1 sweep. 1.0 (no gain) if the sweep has not been run on this node yet."""
    path = RESULTS_DIR / profile / 'summary_v2.csv'
    rates = {}
    for r in read_csv(path) if path.exists() else []:
        if r['output_mode'] == 'fixed' and r['slots'] == '2' and r['req_s']:
            rates[(r['run'], r['concurrency'])] = float(r['req_s'])
    ratios = [rates[(run, '2')] / rates[(run, '1')] for run in RUNS
              if (run, '1') in rates and (run, '2') in rates]
    return median(ratios) or 1.0


def node_model(nodes):
    costs, estimated = load_costs(nodes)
    gains = {n: float(cfg.get('slot_gain') or slot_gain(cfg['profile'])) for n, cfg in nodes.items()}
    return costs, estimated, gains


def print_model(nodes, costs, estimated, gains, mix):
    """Per-node predicted cost per workload and the capacity each node would have for this mix."""
    print('predicted service time when alone, s (* = estimated from the other nodes):')
    short = {'chat': 'chat', 'text': 'txt', 'image': 'img', 'audio': 'aud'}
    head = [short[m] + ('' if m == 'chat' else str(size)) for m, size in RUNS.values()]
    print(f'  {"node":<16}' + ''.join(f'{h:>9}' for h in head) + f'{"gain":>7}{"cap req/s":>11}')
    total = 0.0
    for n in nodes:
        mean = sum(p * costs[n][r] for r, p in mix.items())
        cap = gains[n] / mean   # requests per second the node can sustain on this mix
        total += cap
        cells = ''.join(f'{costs[n][r]:>8.1f}' + ('*' if f'{n}:{r}' in estimated else ' ')
                        for r in RUNS)
        print(f'  {n:<16}{cells}{gains[n]:>7.2f}{cap:>11.3f}')
    print(f'  cluster capacity for this mix ≈ {total:.3f} req/s '
          f'(60% load = {0.6 * total:.3f}, 90% = {0.9 * total:.3f}); gain = slot_gain')
    return total


def capacity(args):
    nodes = load_nodes(args.nodes)
    print_model(nodes, *node_model(nodes), parse_mix(args.mix))


# ---------------------------------------------------------------- routing

class Router:
    """Picks the node for each request on arrival and tracks what every node has been given.

    random: uniform at random (seeded).
    rr:     nodes in nodes.json order, in turn.
    least:  fewest requests in flight per slot (ties broken at random).
    ect:    earliest estimated completion time. Each node's backlog is the predicted work
            (seconds at concurrency 1) it has been given and not yet done; a node works through it
            at slot_gain seconds per second while busy, and it is reset to 0 when the node has
            nothing in flight. A request r finishes on node n at about
                now + (backlog_n + cost_n(r)) / slot_gain_n
            and goes to the node where that is smallest.
    """
    def __init__(self, policy, nodes, costs, gains, seed):
        self.policy = policy
        self.names = list(nodes)
        self.slots = {n: nodes[n]['slots'] for n in nodes}
        self.costs, self.gains = costs, gains
        self.rng = random.Random(seed)
        self.turn = 0
        self.inflight = {n: 0 for n in self.names}
        self.backlog = {n: 0.0 for n in self.names}
        self.updated = {n: 0.0 for n in self.names}
        self.lock = threading.Lock()

    def _backlog(self, n, now):
        self.backlog[n] = max(0.0, self.backlog[n] - self.gains[n] * (now - self.updated[n]))
        self.updated[n] = now
        return self.backlog[n]

    def _finish(self, n, run, now):
        return now + (self._backlog(n, now) + self.costs[n][run]) / self.gains[n]

    def assign(self, run, now):
        """Returns (node, requests in flight there including this one, predicted e2e_s)."""
        with self.lock:
            if self.policy == 'random':
                n = self.rng.choice(self.names)
            elif self.policy == 'rr':
                n = self.names[self.turn % len(self.names)]
                self.turn += 1
            elif self.policy == 'least':
                n = min(self.names, key=lambda n: (self.inflight[n] / self.slots[n], self.rng.random()))
            else:
                n = min(self.names, key=lambda n: (self._finish(n, run, now), self.rng.random()))
            predicted = self._finish(n, run, now) - now   # logged for every policy
            self.backlog[n] += self.costs[n][run]
            self.inflight[n] += 1
            return n, self.inflight[n], predicted

    def done(self, n, now):
        with self.lock:
            self.inflight[n] -= 1
            self._backlog(n, now)
            if self.inflight[n] == 0:
                self.backlog[n] = 0.0


# ---------------------------------------------------------------- replay

def load_datasets(modalities):
    from coco import COCODataset
    from cnn_dailymail import CNNDailyMailDataset
    from dolly import DollyQADataset
    from librispeech import LibriSpeechDataset
    classes = {'chat': DollyQADataset, 'text': CNNDailyMailDataset,
               'image': COCODataset, 'audio': LibriSpeechDataset}
    return {m: classes[m]() for m in modalities}


def warm_up(nodes, datasets, output_mode):
    """One unrecorded request per node and modality (smallest size, input 0), nodes in parallel."""
    first = {}
    for run, (m, size) in RUNS.items():
        if m in datasets:
            first.setdefault(m, size)

    def warm(name):
        with requests.Session() as s:
            for m, size in first.items():
                record = request_once(s, nodes[name]['url'], payload(datasets[m], m, size, 0, output_mode))
                print(f'  warmup {name} {m}: {record["status"]} '
                      + (record['error'] or f'{record["e2e_s"]:.1f}s'), flush=True)
    threads = [threading.Thread(target=warm, args=(n,)) for n in nodes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def run(args):
    nodes = load_nodes(args.nodes)
    trace = read_csv(args.trace)[:args.requests]
    if not trace:
        raise SystemExit(f'{args.trace} has no requests')
    for name, cfg in nodes.items():
        cfg['model'], _, slots, _ = server_info(cfg['url'])
        cfg['slots'] = slots or 1
    costs, estimated, gains = node_model(nodes)
    mix = {r: sum(q['run'] == r for q in trace) / len(trace) for r in RUNS}
    offered = len(trace) / float(trace[-1]['arrival_s'])
    cap = print_model(nodes, costs, estimated, gains, mix)
    print(f'trace {args.trace}: {len(trace)} requests, offered {offered:.3f} req/s '
          f'= {offered / cap:.0%} of predicted capacity', flush=True)

    datasets = load_datasets({q['modality'] for q in trace})
    print('building requests', flush=True)
    bodies = {q['id']: payload(datasets[q['modality']], q['modality'], RUNS[q['run']][1],
                               int(q['input']), args.output) for q in trace}
    if not args.no_warmup:
        warm_up(nodes, datasets, args.output)

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    labels = {'run_id': run_id, 'cluster': args.cluster, 'trace': Path(args.trace).stem,
              'policy': args.policy, 'seed': args.seed, 'output_mode': args.output}
    router = Router(args.policy, nodes, costs, gains, args.seed)
    write_lock = threading.Lock()
    rows = []

    def send(q, node, sent, inflight, predicted):
        with requests.Session() as s:
            record = request_once(s, nodes[node]['url'], bodies[q['id']])
        router.done(node, time.perf_counter() - start)
        row = {**labels, 'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
               'id': q['id'], 'arrival_s': q['arrival_s'], 'sent_s': f'{sent:.3f}', 'node': node,
               'model': nodes[node]['model'], 'node_inflight': inflight,
               'predicted_s': costs[node][q['run']], 'predicted_e2e_s': predicted,
               'run': q['run'], 'modality': q['modality'], 'size': q['size'], 'input': q['input'],
               **record, **derived(record)}
        # Saved as soon as the request finishes, so an interrupted run keeps every finished request.
        with write_lock:
            append_rows(OUT_DIR / 'requests.csv', REQUEST_FIELDS, [row])
            rows.append(row)
        print(f"  {q['id']:>4} {q['run']} {q['modality']}={q['size']} -> {node} "
              f"({inflight} in flight): {record['status']} "
              + (record['error'] or f"{record['e2e_s']:.1f}s, predicted {predicted:.1f}s"), flush=True)

    with ExitStack() as stack:
        monitors = {n: stack.enter_context(RemoteSamples(c['ssh'])) for n, c in nodes.items() if c.get('ssh')}
        print(f'replaying {len(trace)} requests with policy {args.policy} over {list(nodes)}', flush=True)
        threads = []
        start = time.perf_counter()
        for q in trace:
            wait = float(q['arrival_s']) - (time.perf_counter() - start)
            if wait > 0:
                time.sleep(wait)
            now = time.perf_counter() - start
            node, inflight, predicted = router.assign(q['run'], now)
            t = threading.Thread(target=send, args=(q, node, now, inflight, predicted), daemon=True)
            t.start()
            threads.append(t)
        print('all requests sent; waiting for the last ones to finish', flush=True)
        for t in threads:
            t.join()
        makespan = time.perf_counter() - start

    # Summary rows: all requests, each modality, each node.
    price = sum(c.get('price_per_hour') or 0 for c in nodes.values())
    has_price = all(c.get('price_per_hour') for c in nodes.values())
    groups = {'all': rows,
              **{m: [r for r in rows if r['modality'] == m] for m in ('chat', 'text', 'image', 'audio')},
              **{n: [r for r in rows if r['node'] == n] for n in nodes}}
    summaries = []
    for group, members in groups.items():
        if not members:
            continue
        ok = [r for r in members if r['status'] == 'ok']
        valid = lambda field: [r[field] for r in ok if r[field] is not None]
        cpu, ram = monitors[group].metrics() if group in monitors else (None, None)
        summaries.append({
            **labels, 'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'offered_req_s': offered, 'group': group, 'requests': len(members),
            'success': len(ok), 'failed': len(members) - len(ok),
            'median_e2e_s': median(valid('e2e_s')), 'p95_e2e_s': percentile(valid('e2e_s'), .95),
            'p99_e2e_s': percentile(valid('e2e_s'), .99), 'median_ttft_s': median(valid('ttft_s')),
            'p95_ttft_s': percentile(valid('ttft_s'), .95), 'median_queue_s': median(valid('queue_s')),
            'req_s': len(ok) / makespan,
            'usd_per_1k_requests': price * makespan / 3600 / len(ok) * 1000
                                   if group == 'all' and has_price and ok else None,
            'peak_ram_gb': ram, 'cpu_pct': cpu})
    append_rows(OUT_DIR / 'summary.csv', SUMMARY_FIELDS, summaries)
    for s in summaries:
        print(f"  {s['group']:<14} {s['success']}/{s['requests']} ok, median e2e "
              f"{s['median_e2e_s'] or 0:.1f}s, p95 {s['p95_e2e_s'] or 0:.1f}s, "
              f"median TTFT {s['median_ttft_s'] or 0:.1f}s, {s['req_s']:.3f} req/s")
    print(f'Appended results to {OUT_DIR}/requests.csv and summary.csv (run_id={run_id})')


def main():
    parser = argparse.ArgumentParser(description='Phase 2 open-loop load generator and router.')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('trace', help='write a seeded Poisson trace')
    p.add_argument('--rate', type=float, required=True, help='mean arrivals per second')
    p.add_argument('--duration', type=float, default=900, help='seconds of arrivals (default 900)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mix', default=DEFAULT_MIX, help=f'workload mix (default {DEFAULT_MIX})')
    p.add_argument('--out', help='trace file (default traces/rate<r>-dur<d>-seed<s>.csv)')
    p = sub.add_parser('capacity', help='print predicted per-node costs and cluster capacity')
    p.add_argument('--nodes', default='nodes.json')
    p.add_argument('--mix', default=DEFAULT_MIX)
    p = sub.add_parser('run', help='replay a trace over the nodes with one policy')
    p.add_argument('--trace', required=True)
    p.add_argument('--nodes', default='nodes.json')
    p.add_argument('--policy', choices=POLICIES, required=True)
    p.add_argument('--cluster', required=True, help='label for this set of nodes, e.g. A, B, C')
    p.add_argument('--seed', type=int, default=0, help='random policy and tie-breaks')
    p.add_argument('--output', choices=('natural', 'fixed'), default='natural')
    p.add_argument('--requests', type=int, help='replay only the first N requests (quick test)')
    p.add_argument('--no-warmup', action='store_true')
    args = parser.parse_args()
    {'trace': make_trace, 'capacity': capacity, 'run': run}[args.command](args)


if __name__ == '__main__':
    main()
