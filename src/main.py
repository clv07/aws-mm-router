#!/usr/bin/env python3
"""Routing benchmark runner: replays a tracegen.py trace open loop over llama-server nodes (`run`) or a
simulated cluster (`sim`) with router.py's policies, so every policy sees exactly the same traffic.

  python src/main.py capacity --nodes nodes.json                   # predicted cost per node, capacity
  python src/main.py run --trace traces/<name>.csv --nodes nodes.json --policy ect --cluster C
  python src/main.py sim --trace traces/<name>.csv --nodes nodes.json   # every policy, simulated
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import random
import threading
import time

import requests

# Phase 2 measurement: buffer side files without changing request result columns.
from phase2_measurement import Measurements, request_bytes

from benchmark import RUNS, RemoteSamples, append_rows, derived, median, payload, request_once, server_info
from router import (INF, POLICIES, QUEUE_POLICIES, Request, Router, fit_service_model, load_nodes,
                    node_model, read_csv)
from tracegen import DEFAULT_MIX, parse_mix

OUT_DIR = Path("results/routing")
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
SIM_FIELDS = ["run_id", "timestamp_utc", "trace", "policy", "seed", "slo_mult", "group", "requests",
              "median_e2e_s", "p95_e2e_s", "slo_attainment", "req_s", "usd_per_1k_requests",
              "decision_us"]
MODALITIES = ('chat', 'text', 'image', 'audio')


def utc_now():
    """Current UTC time as an ISO timestamp, to the second."""
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def new_run_id():
    """Run id: the current UTC time, e.g. 20261113T184453Z."""
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


# (a) prepare: trace, nodes, requests

def read_trace(path, n):
    """First n requests of a trace (all if n is None)."""
    trace = read_csv(path)[:n]
    if not trace:
        raise SystemExit(f'{path} has no requests')
    return trace


def print_model(nodes, costs, estimated, gains, mix):
    """Print each node's predicted cost per workload and capacity for this mix; return the cluster capacity."""
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
    """`capacity`: print predicted per-node costs and cluster capacity for a mix."""
    nodes = load_nodes(args.nodes)
    print_model(nodes, *node_model(nodes), parse_mix(args.mix))


def load_datasets(modalities):
    """One dataset loader per modality in the trace."""
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


def build_requests(args, trace, nodes):
    """HTTP body for every trace request, then warm up the nodes unless --no-warmup."""
    datasets = load_datasets({q['modality'] for q in trace})
    print('building requests', flush=True)
    bodies = {q['id']: payload(datasets[q['modality']], q['modality'], RUNS[q['run']][1],
                               int(q['input']), args.output) for q in trace}
    if not args.no_warmup:
        warm_up(nodes, datasets, args.output)
    return bodies


def sim_requests(args, trace, nodes, costs, gains, model):
    """Simulator inputs: a Request (prompt tokens, deadline) per trace row and each one's actual alone time per node."""
    # Prompt tokens as logged when this trace ran on the cluster (tokenizing is deterministic, so
    # the router could count them on arrival), else the Phase 1 median.
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
    # Actual alone time drawn once from Phase 1 requests (not the cost model) so every policy sees
    # the same service times; the Phase 1 median if a node has no sample.
    rng = random.Random(args.seed)
    actual = {(r.id, n): rng.choice(model[n]['samples'].get(r.run) or [costs[n][r.run]])
              for r in reqs for n in nodes}
    print(f'simulating {len(reqs)} requests of {args.trace} over '
          + ', '.join(f'{n} ({c["slots"]} slots, gain {gains[n]:.2f})' for n, c in nodes.items())
          + f'; SLO = {args.slo:g} x alone time on the fastest node; '
          f'{sum(q["id"] in logged for q in trace)} prompt token counts from {OUT_DIR}/requests.csv')
    return reqs, actual


# (b) route: live replay and simulator

def replay(args, nodes, trace, costs, gains, bodies, labels):
    """Send each request at its arrival time to the node router.Router picks; returns (rows, makespan, monitors)."""
    # Phase 2 measurement: count serialized payload bytes before the replay clock starts.
    body_bytes = {key: request_bytes(body) for key, body in bodies.items()}
    router = Router(args.policy, nodes, costs, gains, args.seed)
    write_lock = threading.Lock()
    rows = []

    def send(q, node, sent, inflight, predicted):
        with requests.Session() as s:
            record = request_once(s, nodes[node]['url'], bodies[q['id']])
        router.done(node, time.perf_counter() - start)
        measurements.complete(q['id'], record)   # Phase 2 measurement: response bytes, encoder time
        row = {**labels, 'timestamp_utc': utc_now(),
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
        # Phase 2 measurement: skip pre-run samples and flush side files on interruption.
        measurements = Measurements(OUT_DIR, labels['run_id'], monitors, start)
        measurements.thread.start()
        try:
            for q in trace:
                wait = float(q['arrival_s']) - (time.perf_counter() - start)
                if wait > 0:
                    time.sleep(wait)
                now = time.perf_counter() - start
                # Phase 2 measurement: time only assignment to isolate router overhead.
                decision_start = time.perf_counter()
                node, inflight, predicted = router.assign(q['run'], now)
                decision_us = (time.perf_counter() - decision_start) * 1e6
                # Phase 2 measurement: retain payload size and resolved slots for analysis.
                measurements.record(q['id'], body_bytes[q['id']], decision_us, nodes[node]['slots'])
                t = threading.Thread(target=send, args=(q, node, now, inflight, predicted), daemon=True)
                t.start()
                threads.append(t)
            print('all requests sent; waiting for the last ones to finish', flush=True)
            for t in threads:
                t.join()
            makespan = time.perf_counter() - start
        finally:
            # Phase 2 measurement: stop polling after makespan and before monitor teardown.
            measurements.close()
    return rows, makespan, monitors


def run(args):
    """`run`: replay a trace over the live nodes with one policy, then evaluate."""
    nodes = load_nodes(args.nodes)
    trace = read_trace(args.trace, args.requests)
    for cfg in nodes.values():
        cfg['model'], _, slots, _ = server_info(cfg['url'])
        cfg['slots'] = slots or 1
    costs, estimated, gains = node_model(nodes)
    mix = {r: sum(q['run'] == r for q in trace) / len(trace) for r in RUNS}
    offered = len(trace) / float(trace[-1]['arrival_s'])
    cap = print_model(nodes, costs, estimated, gains, mix)
    print(f'trace {args.trace}: {len(trace)} requests, offered {offered:.3f} req/s '
          f'= {offered / cap:.0%} of predicted capacity', flush=True)
    bodies = build_requests(args, trace, nodes)

    labels = {'run_id': new_run_id(), 'cluster': args.cluster, 'trace': Path(args.trace).stem,
              'policy': args.policy, 'seed': args.seed, 'output_mode': args.output}
    rows, makespan, monitors = replay(args, nodes, trace, costs, gains, bodies, labels)
    evaluate_run(rows, nodes, labels, offered, makespan, monitors)


def sim_policy(policy, nodes, costs, gains, model, reqs, actual, seed):
    """Replay reqs on simulated nodes; returns ({id: {node, dispatch, finish}}, decision times in s).

    A node runs up to `slots` requests at once: alone at full speed, k at once at slot_gain / k
    each. Commit-on-arrival policies queue at the node; the slo policy queues at the router."""
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
    """`sim`: simulate the trace with each policy over the node profiles, then evaluate."""
    nodes = load_nodes(args.nodes)
    slots = {k: int(v) for k, _, v in (p.partition('=') for p in args.slots.split(',') if p)}
    for name, cfg in nodes.items():
        cfg['slots'] = slots.get(name, 2)
    costs, _estimated, gains = node_model(nodes)
    model = fit_service_model(nodes)
    trace = read_trace(args.trace, args.requests)
    policies = args.policies.split(',')
    for p in policies:
        if p not in POLICIES + QUEUE_POLICIES:
            raise SystemExit(f'unknown policy {p!r}: use {", ".join(POLICIES + QUEUE_POLICIES)}')
    reqs, actual = sim_requests(args, trace, nodes, costs, gains, model)

    run_id = new_run_id()
    results = [(p, *sim_policy(p, nodes, costs, gains, model, reqs, actual, args.seed)) for p in policies]
    # Phase 2 measurement: export each completed replay before summary evaluation.
    for policy, log, decisions in results:
        # Phase 2 measurement: export service timestamps and sampled solo costs without changing scheduling.
        export_rows = []
        for r in reqs:
            e = log[r.id]
            export_rows.append(dict(source='sim', run_id=run_id, trace=Path(args.trace).stem, policy=policy,
                                    seed=args.seed, slo_mult=args.slo, id=r.id, run=r.run,
                                    modality=r.modality, arrival_s=r.arrival, node=e['node'],
                                    node_slots=nodes[e['node']]['slots'], start_s=e['dispatch'],
                                    finish_s=e['finish'], solo_s=actual[r.id, e['node']],
                                    sim_deadline_s=r.deadline, predicted_finish_s=None))
        # Phase 2 measurement: preserve the optional forecast column as empty; no sim_policy edit.
        append_rows(OUT_DIR / 'sim_requests.csv', list(export_rows[0]), export_rows)
        # Phase 2 measurement: retain every timed call, including dispatches that sent nothing.
        append_rows(OUT_DIR / 'sim_decisions.csv', ['run_id', 'policy', 'n', 'duration_us'],
                    [dict(run_id=run_id, policy=policy, n=i, duration_us=d * 1e6)
                     for i, d in enumerate(decisions)])
    evaluate_sim(results, reqs, nodes, args, run_id)


# (c) evaluate
# EVALUATION ENTRY POINTS: evaluate_run (live `run`) and evaluate_sim (`sim`) turn per-request
# results into the summary rows. Evaluation scripts plug in here.

def percentile(values, q):
    """Linear-interpolated q-quantile (0-1) of values; None if empty."""
    xs = sorted(values)
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def cluster_price(nodes):
    """Total $/hour of the nodes, and whether every node has a price."""
    return (sum(c.get('price_per_hour') or 0 for c in nodes.values()),
            all(c.get('price_per_hour') for c in nodes.values()))


def evaluate_run(rows, nodes, labels, offered, makespan, monitors):
    """Summary rows of a live run (all requests, each modality, each node), appended to summary.csv and printed."""
    price, has_price = cluster_price(nodes)
    groups = {'all': rows,
              **{m: [r for r in rows if r['modality'] == m] for m in MODALITIES},
              **{n: [r for r in rows if r['node'] == n] for n in nodes}}
    summaries = []
    for group, members in groups.items():
        if not members:
            continue
        ok = [r for r in members if r['status'] == 'ok']

        def valid(field):
            return [r[field] for r in ok if r[field] is not None]
        cpu, ram = monitors[group].metrics() if group in monitors else (None, None)
        summaries.append({
            **labels, 'timestamp_utc': utc_now(),
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
    print(f'Appended results to {OUT_DIR}/requests.csv and summary.csv (run_id={labels["run_id"]})')


def evaluate_sim(results, reqs, nodes, args, run_id):
    """Summary rows per policy (all requests, each modality) from [(policy, log, decisions)], appended to sim_summary.csv and printed."""
    stem = Path(args.trace).stem
    price, has_price = cluster_price(nodes)
    summaries = []
    print(f'  {"policy":<7}{"group":<7}{"n":>5}{"p50 s":>9}{"p95 s":>9}{"SLO met":>9}'
          f'{"req/s":>8}{"$/1k":>7}{"decide µs":>11}  share per node')
    for policy, log, decisions in results:
        makespan = max(e['finish'] for e in log.values())
        decide_us = sum(decisions) / len(decisions) * 1e6
        groups = {'all': reqs, **{m: [r for r in reqs if r.modality == m] for m in MODALITIES}}
        for group, members in groups.items():
            if not members:
                continue
            e2e = [log[r.id]['finish'] - r.arrival for r in members]
            met = sum(log[r.id]['finish'] <= r.deadline for r in members) / len(members)
            usd = price * makespan / 3600 / len(reqs) * 1000 if group == 'all' and has_price else None
            share = ' '.join(f'{n.split(".")[0]}={sum(log[r.id]["node"] == n for r in members) / len(members):.0%}'
                             for n in nodes)
            summaries.append({
                'run_id': run_id, 'timestamp_utc': utc_now(),
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
    {'capacity': capacity, 'run': run, 'sim': simulate}[args.command](args)


if __name__ == '__main__':
    main()
