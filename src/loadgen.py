#!/usr/bin/env python3
"""Routing benchmarks: open-loop load generator and router over several llama-server nodes.

  python src/loadgen.py capacity --nodes nodes.json                   # predicted cost per node, capacity
  python src/loadgen.py run --trace traces/<name>.csv --nodes nodes.json --policy ect --cluster C

Traces are made with tracegen.py. A trace fixes every request's arrival time, workload (benchmark.py RUNS) and input, so every
policy and cluster replays exactly the same traffic. Each request is sent at its arrival time
whether or not earlier ones have finished (open loop). The policy picks the node on arrival;
the request then waits in that node's llama-server queue if all of its slots are busy.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import threading
import time

import requests

from benchmark import (RESULTS_DIR, RUNS, RemoteSamples, append_rows, derived, median, payload,
                       request_once, server_info)

OUT_DIR = Path("results/routing")
DEFAULT_MIX = "chat=50,text=20,image=15,audio=15"   # percent per modality; sizes uniform within it
POLICIES = ("random", "rr", "least", "ect")
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
    """Median e2e_s per workload at concurrency 1 in one Phase 1 results folder, natural output
    if measured, otherwise fixed (128 tokens, so somewhat longer). summary_v2.csv rows replace
    summary.csv rows and later rows replace earlier ones; workloads with any failed request
    are left out."""
    costs = {'natural': {}, 'fixed': {}}
    for file in ('summary.csv', 'summary_v2.csv'):
        path = RESULTS_DIR / profile / file
        for r in read_csv(path) if path.exists() else []:
            if r.get('concurrency') == '1' and r['failed'] == '0' and r['median_e2e_s']:
                costs[r.get('output_mode', 'natural')][r['run']] = float(r['median_e2e_s'])
    return {**costs['fixed'], **costs['natural']}


def load_costs(nodes):
    """Predicted service time (s) of each workload on each node when it runs alone, from the
    node's Phase 1 results (profile_costs). A workload not measured on a node (e.g. failed audio) is estimated
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


def phase1_requests(profile):
    """Successful concurrency-1 requests (both output modes) in one Phase 1 results folder."""
    rows = []
    for file in ('requests.csv', 'requests_v2.csv'):
        path = RESULTS_DIR / profile / file
        rows += [r for r in (read_csv(path) if path.exists() else [])
                 if r.get('concurrency') == '1' and r['status'] == 'ok' and r['e2e_s']
                 and r['prompt_tokens'] and r['output_tokens']]
    return rows


def fit_service_model(nodes):
    """Token-based service time per node, S(r, n) = a[n, m] + b[n, m] * prompt_tokens
    + d[n] * est_output_tokens, fit on the node's Phase 1 concurrency-1 requests. d is the median
    time per output token (tpot); a and b are a least-squares line per modality through
    e2e - d * output_tokens. Also keeps, per workload, the median prompt and natural output
    length, the natural e2e samples (for the simulator), and the workloads that failed (won't fit)."""
    model = {}
    for name, cfg in nodes.items():
        rows = phase1_requests(cfg['profile'])
        if not rows:
            raise SystemExit(f'node {name!r}: no concurrency-1 requests in {cfg["profile"]}')
        d = median([float(r['tpot_ms']) / 1000 for r in rows if r['tpot_ms']])
        a, b = {}, {}
        for m in {r['modality'] for r in rows}:
            xs = [float(r['prompt_tokens']) for r in rows if r['modality'] == m]
            ys = [float(r['e2e_s']) - d * float(r['output_tokens']) for r in rows if r['modality'] == m]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            var = sum((x - mx) ** 2 for x in xs)
            b[m] = max(0.0, sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var) if var else 0.0
            a[m] = my - b[m] * mx
        natural = [r for r in rows if r['output_mode'] == 'natural']
        failed = set()
        for file in ('summary.csv', 'summary_v2.csv'):
            path = RESULTS_DIR / cfg['profile'] / file
            failed |= {r['run'] for r in (read_csv(path) if path.exists() else [])
                       if r.get('concurrency') == '1' and r['failed'] not in ('', '0')}
        model[name] = {
            'a': a, 'b': b, 'd': d, 'infeasible': failed,
            'prompt': {run: median([float(r['prompt_tokens']) for r in rows if r['run'] == run])
                       for run in RUNS if any(r['run'] == run for r in rows)},
            'out': {run: median([float(r['output_tokens']) for r in natural if r['run'] == run])
                    for run in RUNS if any(r['run'] == run for r in natural)},
            'samples': {run: [float(r['e2e_s']) for r in natural if r['run'] == run]
                        for run in RUNS if any(r['run'] == run for r in natural)}}
    return model


# ---------------------------------------------------------------- routing

@dataclass(eq=False)
class Request:
    id: int
    run: str
    modality: str
    prompt_tokens: float
    arrival: float
    deadline: float


# Defaults of the full method (SLO-aware Sufferage); the ect baseline is
# dict(service='median', slowdown='ect', wait='backlog', correction=False, feasibility=False).
COST_DEFAULTS = dict(service='token', slowdown='fixed', wait='slot', correction=True, feasibility=True)
DECISION_DEFAULTS = dict(order='slack', tiebreak='sufferage', aging=0.0, wait_for_best=True)


class Router:
    """Picks the node for each request and tracks what every node has been given.

    Commit on arrival (assign/done):
    random: uniform at random (seeded).
    rr:     nodes in nodes.json order, in turn.
    least:  fewest requests in flight per slot (ties broken at random).
    ect:    earliest estimated completion time. Each node's backlog is the predicted work
            (seconds at concurrency 1) it has been given and not yet done; a node works through it
            at slot_gain seconds per second while busy, and it is reset to 0 when the node has
            nothing in flight. A request r finishes on node n at about
                now + (backlog_n + cost_n(r)) / slot_gain_n
            and goes to the node where that is smallest.

    Router queue (dispatch), requests wait here until a slot frees:
    slo:    SLO-aware Sufferage. cost_model() gives T(r, n) = now + W_n + S(r, n) * f_n(k) * c_n;
            pick() sends the queued request with the least slack (deadline - best T, ties to the
            largest sufferage, second-best T - best T) to its best node.
    """
    def __init__(self, policy, nodes, costs, gains, seed, model=None, cost=None, decision=None):
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
        # Router queue and node state.
        self.model = model
        self.cost = {**COST_DEFAULTS, **(cost or {})}
        self.decision = {**DECISION_DEFAULTS, **(decision or {})}
        self.queue = []
        self.running = {n: {} for n in self.names}     # request id -> predicted finish
        self.started = {}                              # request id -> (dispatch time, predicted S * f)
        self.healthy = {n: True for n in self.names}
        self.mem_limit = {n: model[n]['infeasible'] if model else set() for n in self.names}
        self.c = {n: 1.0 for n in self.names}

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

    def cost_model(self, r, n, now, service='token', slowdown='fixed', wait='slot',
                   correction=True, feasibility=True):
        """
        Returns (T, S * f): predicted finish time T(r, n) = now + W_n + S(r, n) * f_n(k) * c_n,
        and the uncorrected run time S * f. (inf, inf) if request r can't run on node n.
          W_n     wait: time until r gets a slot on n
          S(r, n) service: time r takes on n when running alone
          f_n(k)  slowdown: how much slower r runs while sharing n with k-1 others
          c_n     correction: learned ratio of actual to predicted run time on n
        """
        # feasibility: node is healthy and r fits in its memory
        if feasibility and (not self.healthy[n] or r.run in self.mem_limit[n]):
            return INF, INF

        fit = self.model[n] if self.model else {}
        n_running = len(self.running[n])
        has_free_slot = n_running < self.slots[n]
        
        # S: token-based fit, or the Phase 1 median if there is no fit
        service_time = (fit['a'][r.modality] + fit['b'][r.modality] * r.prompt_tokens
                        + fit['d'] * fit['out'][r.run]
                        if service == 'token' and r.modality in fit['a'] and r.run in fit['out']
                        else self.costs[n][r.run])
        
        # f: k = requests sharing n once r starts
        k = n_running + 1 if has_free_slot else self.slots[n]
        slowdown_factor = (1 / self.gains[n] if slowdown == 'ect'
                           else 1.0 if k == 1 else k / self.gains[n])
        
        # W: ect backlog, or until the first running request finishes if n is full
        wait_time = (self._backlog(n, now) / self.gains[n] if wait == 'backlog'
                     else max(0.0, min(self.running[n].values()) - now)
                     if wait == 'slot' and not has_free_slot else 0.0)
        
        correction_factor = self.c[n] if correction else 1.0
        run_time = service_time * slowdown_factor

        return now + wait_time + run_time * correction_factor, run_time

    # decision rule
    def pick(self, now, order='slack', tiebreak='sufferage', aging=0.0, wait_for_best=True):
        """
        Select node to run the request.
        Most urgent (smallest priority) queued request that can start now, with its target node.
        """
        best = None
        for r in self.queue:
            ts = sorted((self.cost_model(r, n, now, **self.cost)[0], i, n)
                        for i, n in enumerate(self.names))
            best_t, second_t = ts[0][0], ts[1][0] if len(ts) > 1 else INF

            # doesn't fit in any node now
            if best_t == INF: 
                continue
            
            # sufferage: how much later r finishes if it misses its best node
            sufferage = second_t - best_t
            
            # order: 
            # slack = time to spare before the deadline
            # sufferage = most to lose first
            # fifo = earliest arrival first
            priority = {'slack': r.deadline - best_t, 'sufferage': -sufferage,
                        'fifo': r.arrival}[order]
            
            # aging: the longer r waits, the more urgent it gets
            priority -= aging * (now - r.arrival)
            
            # wait_for_best: only the best node, even if busy; else the best node with a free slot
            free = [n for t, _, n in ts if t < INF and len(self.running[n]) < self.slots[n]]
            target = (ts[0][2] if ts[0][2] in free else None) if wait_for_best else \
                     (free[0] if free else None)
            if target is None:
                continue
            
            # tiebreak: equal priority goes to the larger sufferage, then the earlier arrival
            key = (priority, -sufferage if tiebreak == 'sufferage' else 0.0, r.arrival)
            if best is None or key < best[0]:
                best = (key, r, target)
        return best[1:] if best else None

    # queue
    def dispatch(self, now, arrived=None, finished=None):
        """Handles one event, then sends queued requests to free slots while pick() finds one.
        arrived: a new request. 
        finished: (request, node, seconds from dispatch to finish, ok).
        Returns [(request, node)] for the caller to forward (HTTP or simulator).
        """
        if arrived:
            self.queue.append(arrived)
        if finished:
            r, n, actual, ok = finished
            del self.running[n][r.id]
            _, predicted = self.started.pop(r.id)
            if not ok:   # a failed request marks the node unhealthy
                self.healthy[n] = False
            elif predicted > 0:   # c_n: moving average of actual / predicted run time
                self.c[n] = 0.9 * self.c[n] + 0.1 * actual / predicted
            self._backlog(n, now)
            if not self.running[n]:
                self.backlog[n] = 0.0
        out = []
        while (choice := self.pick(now, **self.decision)) is not None:
            r, n = choice
            self.queue.remove(r)
            self.running[n][r.id], sf = self.cost_model(r, n, now, **{**self.cost, 'feasibility': False})
            self.started[r.id] = (now, sf)
            self.backlog[n] += self.costs[n][r.run]
            out.append((r, n))
        return out


# ---------------------------------------------------------------- replay

def load_datasets(modalities):
    from loaders.coco import COCODataset
    from loaders.cnn_dailymail import CNNDailyMailDataset
    from loaders.dolly import DollyQADataset
    from loaders.librispeech import LibriSpeechDataset
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


# ---------------------------------------------------------------- simulation

SIM_FIELDS = ["run_id", "timestamp_utc", "trace", "policy", "seed", "slo_mult", "group", "requests",
              "median_e2e_s", "p95_e2e_s", "slo_attainment", "req_s", "usd_per_1k_requests",
              "decision_us"]


def sim_policy(policy, nodes, costs, gains, model, reqs, actual, seed):
    """Replays reqs on simulated nodes; returns ({id: {node, dispatch, finish}}, decision times in s).
    A node runs up to `slots` requests at once: alone at full speed, k at once at slot_gain / k
    each. Commit-on-arrival policies wait in the node's own FIFO queue when its slots are busy;
    the slo policy waits in the router queue."""
    router = Router(policy, nodes, costs, gains, seed, model)
    running = {n: {} for n in nodes}   # id -> remaining seconds of work at full speed
    waiting = {n: deque() for n in nodes}
    by_id = {r.id: r for r in reqs}
    arrivals = deque(sorted(reqs, key=lambda r: r.arrival))
    log, decisions, now = {}, [], 0.0

    def rate(n):
        return 1.0 if len(running[n]) <= 1 else gains[n] / len(running[n])

    def start(r, n):
        running[n][r.id] = actual[r.id, n]
        log[r.id] = {'node': n, 'dispatch': now}

    def timed(fn, *a):
        t0 = time.perf_counter()
        out = fn(*a)
        decisions.append(time.perf_counter() - t0)
        return out

    while arrivals or any(running.values()):
        finish, who = INF, None
        for n in nodes:
            if running[n]:
                i = min(running[n], key=running[n].get)
                t = now + running[n][i] / rate(n)
                if t < finish:
                    finish, who = t, (n, i)
        arrival = arrivals[0].arrival if arrivals else INF
        t = min(finish, arrival)
        for n in nodes:
            k = rate(n) * (t - now)
            for i in running[n]:
                running[n][i] -= k
        now = t
        if finish <= arrival:
            n, i = who
            del running[n][i]
            log[i]['finish'] = now
            if policy in POLICIES:
                router.done(n, now)
                if waiting[n]:
                    start(waiting[n].popleft(), n)
            else:
                for r, m in timed(router.dispatch, now, None,
                                  (by_id[i], n, now - log[i]['dispatch'], True)):
                    start(r, m)
        else:
            r = arrivals.popleft()
            if policy in POLICIES:
                n = timed(router.assign, r.run, now)[0]
                if len(running[n]) < nodes[n]['slots']:
                    start(r, n)
                else:
                    waiting[n].append(r)
            else:
                for q, m in timed(router.dispatch, now, r):
                    start(q, m)
    return log, decisions


def simulate(args):
    nodes = load_nodes(args.nodes)
    slots = {k: int(v) for k, _, v in (p.partition('=') for p in args.slots.split(',') if p)}
    for name, cfg in nodes.items():
        cfg['slots'] = slots.get(name, 2)
    costs, estimated, gains = node_model(nodes)
    model = fit_service_model(nodes)
    trace = read_csv(args.trace)[:args.requests]
    if not trace:
        raise SystemExit(f'{args.trace} has no requests')
    policies = args.policies.split(',')
    for p in policies:
        if p not in POLICIES + QUEUE_POLICIES:
            raise SystemExit(f'unknown policy {p!r}: use {", ".join(POLICIES + QUEUE_POLICIES)}')

    # Prompt tokens per request: as logged when this trace ran on the cluster (tokenizing is
    # deterministic, so the router could count them on arrival), else the Phase 1 median.
    stem = Path(args.trace).stem
    logged = {}
    if (OUT_DIR / 'requests.csv').exists():
        for r in read_csv(OUT_DIR / 'requests.csv'):
            if r['trace'] == stem and r['status'] == 'ok' and r['prompt_tokens']:
                logged.setdefault(r['id'], float(r['prompt_tokens']))
    first = model[next(iter(nodes))]
    reqs = []
    for q in trace:
        arrival = float(q['arrival_s'])
        slo = args.slo * min(costs[n][q['run']] for n in nodes)   # x alone time on the fastest node
        reqs.append(Request(int(q['id']), q['run'], q['modality'],
                            logged.get(q['id'], first['prompt'].get(q['run'], 0.0)),
                            arrival, arrival + slo))
    # Actual time alone, drawn once from Phase 1 requests (not from the cost model) so every
    # policy sees the same service times; the Phase 1 median if a node has no sample.
    rng = random.Random(args.seed)
    actual = {(r.id, n): rng.choice(model[n]['samples'].get(r.run) or [costs[n][r.run]])
              for r in reqs for n in nodes}
    print(f'simulating {len(reqs)} requests of {args.trace} over '
          + ', '.join(f'{n} ({c["slots"]} slots, gain {gains[n]:.2f})' for n, c in nodes.items())
          + f'; SLO = {args.slo:g} x alone time on the fastest node; '
          f'{sum(q["id"] in logged for q in trace)} prompt token counts from {OUT_DIR}/requests.csv')

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    price = sum(c.get('price_per_hour') or 0 for c in nodes.values())
    has_price = all(c.get('price_per_hour') for c in nodes.values())
    summaries = []
    print(f'  {"policy":<7}{"group":<7}{"n":>5}{"p50 s":>9}{"p95 s":>9}{"SLO met":>9}'
          f'{"req/s":>8}{"$/1k":>7}{"decide µs":>11}  share per node')
    for policy in policies:
        log, decisions = sim_policy(policy, nodes, costs, gains, model, reqs, actual, args.seed)
        makespan = max(e['finish'] for e in log.values())
        decide_us = sum(decisions) / len(decisions) * 1e6
        groups = {'all': reqs, **{m: [r for r in reqs if r.modality == m]
                                  for m in ('chat', 'text', 'image', 'audio')}}
        for group, members in groups.items():
            if not members:
                continue
            e2e = [log[r.id]['finish'] - r.arrival for r in members]
            met = sum(log[r.id]['finish'] <= r.deadline for r in members) / len(members)
            usd = price * makespan / 3600 / len(reqs) * 1000 if group == 'all' and has_price else None
            share = ' '.join(f'{n.split(".")[0]}={sum(log[r.id]["node"] == n for r in members) / len(members):.0%}'
                             for n in nodes)
            summaries.append({
                'run_id': run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'trace': stem, 'policy': policy, 'seed': args.seed, 'slo_mult': args.slo,
                'group': group, 'requests': len(members), 'median_e2e_s': median(e2e),
                'p95_e2e_s': percentile(e2e, .95), 'slo_attainment': met,
                'req_s': len(members) / makespan if group == 'all' else None,
                'usd_per_1k_requests': usd, 'decision_us': decide_us if group == 'all' else None})
            print(f'  {policy if group == "all" else "":<7}{group:<7}{len(members):>5}'
                  f'{median(e2e):>9.1f}{percentile(e2e, .95):>9.1f}{met:>9.0%}'
                  + (f'{len(members) / makespan:>8.3f}{usd or 0:>7.2f}{decide_us:>11.1f}'
                     if group == 'all' else ' ' * 26) + f'  {share}')
    append_rows(OUT_DIR / 'sim_summary.csv', SIM_FIELDS, summaries)
    print(f'Appended results to {OUT_DIR}/sim_summary.csv (run_id={run_id})')


def main():
    parser = argparse.ArgumentParser(description='Open-loop load generator and router for the routing benchmarks.')
    sub = parser.add_subparsers(dest='command', required=True)
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
    p = sub.add_parser('sim', help='simulate a trace over the nodes with every policy')
    p.add_argument('--trace', required=True)
    p.add_argument('--nodes', default='nodes.json')
    p.add_argument('--policies', default='rr,least,ect,slo',
                   help='comma-separated, from ' + ', '.join(POLICIES + QUEUE_POLICIES))
    p.add_argument('--slots', default='c7i.xlarge=1',
                   help='slots per node as name=k,... (default 2 for nodes not listed)')
    p.add_argument('--slo', type=float, default=5.0,
                   help='SLO as a multiple of the alone time on the fastest node (default 5)')
    p.add_argument('--seed', type=int, default=0, help='service time draws and tie-breaks')
    p.add_argument('--requests', type=int, help='simulate only the first N requests')
    args = parser.parse_args()
    {'capacity': capacity, 'run': run}[args.command](args)


if __name__ == '__main__':
    main()
