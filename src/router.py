#!/usr/bin/env python3
"""Routing algorithm: the Router (policies random, rr, least, ect, slo), its cost model, and the node
models it is built from (Phase 1 profiles + nodes.json). Library only; main.py and tracegen.py use it.

  python src/main.py sim --trace traces/<name>.csv --nodes nodes.json      # every policy, simulated
  from router import Router, load_nodes, node_model, fit_service_model    # in other scripts
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
import threading

from benchmark import RESULTS_DIR, RUNS, median

POLICIES = ("random", "rr", "least", "ect")      # commit on arrival; `run` and `sim`
QUEUE_POLICIES = ("slo",)                         # router queue, dispatch when a slot frees; `sim` only
INF = float('inf')

# Defaults of the full method (SLO-aware Sufferage); the ect baseline is
# dict(service='median', slowdown='ect', wait='backlog', correction=False, feasibility=False).
COST_DEFAULTS = dict(service='token', slowdown='fixed', wait='slot', correction=True, feasibility=True)
DECISION_DEFAULTS = dict(order='slack', tiebreak='sufferage', aging=0.0, wait_for_best=True)


def read_csv(path):
    """All rows of a CSV file as dicts."""
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


### node models from Phase 1 ###

def load_nodes(path):
    """Read nodes.json ({name: {url, profile, ssh?, price_per_hour?, slot_gain?}}); entry order = rr order."""
    nodes = json.loads(Path(path).read_text())
    for name, cfg in nodes.items():
        for key in ('url', 'profile'):
            if key not in cfg:
                raise SystemExit(f'{path}: node {name!r} has no "{key}"')
        if not (RESULTS_DIR / cfg['profile']).is_dir():
            raise SystemExit(f'{path}: node {name!r}: no folder {RESULTS_DIR / cfg["profile"]}')
    return nodes


def profile_rows(profile, *files):
    """Rows of the given CSV files in one Phase 1 results folder, in file order; missing files are skipped."""
    paths = [RESULTS_DIR / profile / file for file in files]
    return [r for path in paths if path.exists() for r in read_csv(path)]


def profile_costs(profile):
    """Median concurrency-1 e2e_s per workload in one profile, natural output if measured, else fixed.

    summary_v2.csv rows replace summary.csv rows, later rows replace earlier ones, and workloads
    with any failed request are left out."""
    costs = {'natural': {}, 'fixed': {}}
    for r in profile_rows(profile, 'summary.csv', 'summary_v2.csv'):
        if r.get('concurrency') == '1' and r['failed'] == '0' and r['median_e2e_s']:
            costs[r.get('output_mode', 'natural')][r['run']] = float(r['median_e2e_s'])
    return {**costs['fixed'], **costs['natural']}


def load_costs(nodes):
    """Predicted alone service time (s) per node and workload, plus the 'node:run' cells that were estimated.

    A workload a node lacks is estimated from every other Phase 1 folder that has it (that folder's
    time x this node's median ratio to it on shared workloads); the median estimate is used."""
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
    """Median req_s at concurrency 2 / concurrency 1 (2 slots, fixed output) in the profile; 1.0 if not measured."""
    rates = {(r['run'], r['concurrency']): float(r['req_s'])
             for r in profile_rows(profile, 'summary_v2.csv')
             if r['output_mode'] == 'fixed' and r['slots'] == '2' and r['req_s']}
    ratios = [rates[(run, '2')] / rates[(run, '1')] for run in RUNS
              if (run, '1') in rates and (run, '2') in rates]
    return median(ratios) or 1.0


def node_model(nodes):
    """Alone costs, estimated cells and slot gain per node (nodes.json `slot_gain` overrides the profile)."""
    costs, estimated = load_costs(nodes)
    gains = {n: float(cfg.get('slot_gain') or slot_gain(cfg['profile'])) for n, cfg in nodes.items()}
    return costs, estimated, gains


def phase1_requests(profile):
    """Successful concurrency-1 requests (both output modes) in one Phase 1 results folder."""
    return [r for r in profile_rows(profile, 'requests.csv', 'requests_v2.csv')
            if r.get('concurrency') == '1' and r['status'] == 'ok' and r['e2e_s']
            and r['prompt_tokens'] and r['output_tokens']]


def per_run(rows, field):
    """Float values of one field per workload, in RUNS order, for workloads that have rows."""
    return {run: values for run in RUNS
            if (values := [float(r[field]) for r in rows if r['run'] == run])}


def fit_service_model(nodes):
    """Token-based alone time per node, S(r, n) = a[n, m] + b[n, m] * prompt_tokens + d[n] * est_output_tokens.

    d is the median time per output token; a, b are a least-squares line per modality through
    e2e - d * output_tokens. Also keeps per-workload median prompt and natural output lengths,
    natural e2e samples (for the simulator) and the workloads that failed (won't fit in memory)."""
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
        failed = {r['run'] for r in profile_rows(cfg['profile'], 'summary.csv', 'summary_v2.csv')
                  if r.get('concurrency') == '1' and r['failed'] not in ('', '0')}
        model[name] = {
            'a': a, 'b': b, 'd': d, 'infeasible': failed,
            'prompt': {run: median(v) for run, v in per_run(rows, 'prompt_tokens').items()},
            'out': {run: median(v) for run, v in per_run(natural, 'output_tokens').items()},
            'samples': per_run(natural, 'e2e_s')}
    return model


### routing ###

@dataclass(eq=False)
class Request:
    id: int
    run: str
    modality: str
    prompt_tokens: float
    arrival: float
    deadline: float


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
        # Router queue and node state (slo policy).
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
        """Drain node n's backlog up to now and return it."""
        self.backlog[n] = max(0.0, self.backlog[n] - self.gains[n] * (now - self.updated[n]))
        self.updated[n] = now
        return self.backlog[n]

    def _finish(self, n, run, now):
        """ect-style predicted finish time of workload run on node n."""
        return now + (self._backlog(n, now) + self.costs[n][run]) / self.gains[n]

    def assign(self, run, now):
        """Commit a request on arrival; returns (node, requests in flight there including it, predicted e2e_s)."""
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
        """Record that a request committed with assign() finished on node n."""
        with self.lock:
            self.inflight[n] -= 1
            self._backlog(n, now)
            if self.inflight[n] == 0:
                self.backlog[n] = 0.0

    def cost_model(self, r, n, now, service='token', slowdown='fixed', wait='slot',
                   correction=True, feasibility=True):
        """Return (T, S * f): predicted finish T(r, n) = now + W_n + S(r, n) * f_n(k) * c_n and the uncorrected run time.

        W_n wait for a slot, S alone service time, f_n(k) slowdown with k sharing n, c_n learned
        actual / predicted ratio. (inf, inf) if r can't run on n."""
        # Feasibility: node is healthy and r fits in its memory.
        if feasibility and (not self.healthy[n] or r.run in self.mem_limit[n]):
            return INF, INF

        fit = self.model[n] if self.model else {}
        n_running = len(self.running[n])
        has_free_slot = n_running < self.slots[n]

        # S: token-based fit, or the Phase 1 median if there is no fit.
        service_time = (fit['a'][r.modality] + fit['b'][r.modality] * r.prompt_tokens
                        + fit['d'] * fit['out'][r.run]
                        if service == 'token' and r.modality in fit['a'] and r.run in fit['out']
                        else self.costs[n][r.run])

        # f: k = requests sharing n once r starts.
        k = n_running + 1 if has_free_slot else self.slots[n]
        slowdown_factor = (1 / self.gains[n] if slowdown == 'ect'
                           else 1.0 if k == 1 else k / self.gains[n])

        # W: ect backlog, or until the first running request finishes if n is full.
        wait_time = (self._backlog(n, now) / self.gains[n] if wait == 'backlog'
                     else max(0.0, min(self.running[n].values()) - now)
                     if wait == 'slot' and not has_free_slot else 0.0)

        correction_factor = self.c[n] if correction else 1.0
        run_time = service_time * slowdown_factor
        return now + wait_time + run_time * correction_factor, run_time

    def pick(self, now, order='slack', tiebreak='sufferage', aging=0.0, wait_for_best=True):
        """Decision rule: the most urgent queued request that can start now, with its target node (or None)."""
        best = None
        for r in self.queue:
            ts = sorted((self.cost_model(r, n, now, **self.cost)[0], i, n)
                        for i, n in enumerate(self.names))
            best_t, second_t = ts[0][0], ts[1][0] if len(ts) > 1 else INF
            if best_t == INF:   # fits on no node now
                continue

            # Sufferage: how much later r finishes if it misses its best node.
            sufferage = second_t - best_t
            # order: slack = time to spare before the deadline, sufferage = most to lose, fifo = arrival.
            priority = {'slack': r.deadline - best_t, 'sufferage': -sufferage,
                        'fifo': r.arrival}[order]
            # Aging: the longer r waits, the more urgent it gets.
            priority -= aging * (now - r.arrival)

            # wait_for_best: only the best node, even if busy; else the best node with a free slot.
            free = [n for t, _, n in ts if t < INF and len(self.running[n]) < self.slots[n]]
            if wait_for_best:
                target = ts[0][2] if ts[0][2] in free else None
            else:
                target = free[0] if free else None
            if target is None:
                continue

            # Tiebreak: equal priority goes to the larger sufferage, then the earlier arrival.
            key = (priority, -sufferage if tiebreak == 'sufferage' else 0.0, r.arrival)
            if best is None or key < best[0]:
                best = (key, r, target)
        return best[1:] if best else None

    def dispatch(self, now, arrived=None, finished=None):
        """Handle one event, then send queued requests to free slots while pick() finds one.

        arrived: a new Request; finished: (request, node, seconds from dispatch to finish, ok).
        Returns [(request, node)] for the caller to forward (HTTP or simulator)."""
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
