#!/usr/bin/env python3
"""Phase 2 traffic generator: traces for loadgen.py with steady, bursty, step, ramp or replayed
arrivals, named workload mixes, and per-request deadlines.

  python src/tracegen.py make --arrivals poisson --load 0.9 --nodes nodes.json --mix default --seed 1
  python src/tracegen.py make --arrivals onoff --rate 0.13 --burst-factor 3 --burst-fraction 0.2 --mix image-heavy
  python src/tracegen.py make --arrivals replay --replay times.csv --replay-column timestamp --rate 0.13
  python src/tracegen.py describe traces/phase2/*.csv

A trace is a CSV that `loadgen.py run` replays unchanged (it reads id, arrival_s, run, modality,
size and input, and ignores the other columns), plus a JSON file next to it with every parameter
and summary statistics, so each result can be traced back to how its traffic was made.

Comparability. Every arrival process is a time-rescaling of the same unit-rate Poisson stream,
and workloads are drawn from a separate random stream in arrival order. So with one seed:
- request k has the same workload and input in every trace (steady, bursty, any load);
- traces differ only in when requests arrive.
Rates are mean rates over the whole trace for poisson and onoff (bursty traces are rescaled so
their mean matches exactly); for step and ramp they are the base rate the shape is applied to.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path
import random
import statistics

from loadgen import RUNS, TRACES_DIR, node_model, load_nodes, parse_mix

MIXES = {   # percent per modality; sizes share their modality's weight equally
    'default':      'chat=50,text=20,image=15,audio=15',   # the routing benchmarks' mix
    'uniform':      'chat=25,text=25,image=25,audio=25',
    'chat-heavy':   'chat=80,text=10,image=5,audio=5',
    'text-heavy':   'chat=20,text=60,image=10,audio=10',
    'image-heavy':  'chat=20,text=10,image=60,audio=10',
    'audio-heavy':  'chat=20,text=10,image=10,audio=60',
}
SLO_DEFAULT = 'chat=3,text=4,image=4,audio=4'   # deadline = factor x fastest solo time of that workload
FIELDS = ['id', 'arrival_s', 'run', 'modality', 'size', 'input', 'slo_factor', 'deadline_s', 'burst']
SEED_STREAMS = {'arrivals': 0, 'workloads': 1, 'bursts': 2}   # independent random streams per seed


# ---------------------------------------------------------------- rate shapes
# A shape is a list of (start_s, end_s, rate) segments covering [0, duration], plus a flag per
# segment saying whether it is a burst. Arrivals are the unit-rate stream mapped through the
# inverse of the cumulative rate.

def poisson_shape(rate, duration, _args, _rng):
    return [(0.0, duration, rate, False)]


def onoff_shape(rate, duration, args, rng):
    """Two-state bursty process: exponential on (burst) and off periods. During a burst the rate is
    burst_factor x the mean; between bursts it is lower, so the mean stays `rate`."""
    p, b = args.burst_fraction, args.burst_factor
    if not 0 < p < 1 or b <= 1 or p * b >= 1:
        raise SystemExit('onoff needs 0 < --burst-fraction < 1, --burst-factor > 1 and '
                         'burst-fraction x burst-factor < 1 (the rate between bursts must stay >= 0)')
    on_mean = args.burst_length
    off_mean = on_mean * (1 - p) / p
    hi, lo = b * rate, rate * (1 - p * b) / (1 - p)
    segments, t, on = [], 0.0, rng.random() < p
    while t < duration:
        length = rng.expovariate(1 / (on_mean if on else off_mean))
        end = min(duration, t + length)
        segments.append((t, end, hi if on else lo, on))
        t, on = end, not on
    # Rescale so this trace's mean rate is exactly `rate` (bursty and steady then carry the same load).
    total = sum((e - s) * r for s, e, r, _ in segments)
    scale = rate * duration / total if total else 1.0
    return [(s, e, r * scale, on) for s, e, r, on in segments]


def step_shape(rate, duration, args, _rng):
    """Flash crowd: `rate`, then step_factor x rate for step_length seconds from step_at."""
    start, end = args.step_at, min(duration, args.step_at + args.step_length)
    parts = [(0.0, start, rate, False), (start, end, rate * args.step_factor, True),
             (end, duration, rate, False)]
    return [p for p in parts if p[1] > p[0]]


def ramp_shape(rate, duration, args, _rng):
    """Load that rises (or falls) linearly from ramp_from x rate to ramp_to x rate, in 1 s steps."""
    steps = max(1, int(math.ceil(duration)))
    width = duration / steps
    return [(i * width, (i + 1) * width,
             rate * (args.ramp_from + (args.ramp_to - args.ramp_from) * (i + 0.5) / steps), False)
            for i in range(steps)]


SHAPES = {'poisson': poisson_shape, 'onoff': onoff_shape, 'step': step_shape, 'ramp': ramp_shape}


def arrivals_from_shape(segments, rng):
    """Arrival times of an inhomogeneous Poisson process with the given piecewise-constant rate."""
    starts, cum, total = [], [], 0.0
    for s, e, r, _ in segments:
        starts.append(total)
        total += (e - s) * r
        cum.append(total)
    times, unit = [], 0.0
    while True:
        unit += rng.expovariate(1.0)
        if unit > total:
            return times
        i = bisect.bisect_left(cum, unit)
        s, e, r, burst = segments[i]
        times.append((s + (unit - starts[i]) / r if r else s, burst))


def arrivals_from_replay(path, column, unit, rate, duration, offset):
    """Arrival times from a real trace: rate x duration consecutive arrivals, starting at the
    `offset`-th, shifted to start at 0 and stretched or squeezed to fill `duration`. Keeps the real
    trace's burstiness and ordering at the mean rate our cluster can take (a real service is
    usually far busier). Different offsets give different windows of the same trace."""
    factor = {'s': 1.0, 'ms': 1e-3, 'us': 1e-6}[unit]
    with open(path, newline='', encoding='utf-8') as f:
        raw = sorted(float(r[column]) * factor for r in csv.DictReader(f) if r.get(column))
    count = round(rate * duration)
    window = raw[offset:offset + count + 1]   # one extra arrival marks where the window ends
    if len(window) < count + 1:
        raise SystemExit(f'{path}: {len(raw)} arrivals in column {column!r}; need {offset + count + 1} '
                         f'for --rate {rate:g} over {duration:g} s from --replay-offset {offset}')
    scale = duration / (window[-1] - window[0])
    return [((t - window[0]) * scale, False) for t in window[:-1]]


# ---------------------------------------------------------------- trace

def best_costs(nodes_path):
    """Fastest solo time of every workload on any node, and the cluster's capacity per mix."""
    nodes = load_nodes(nodes_path)
    costs, _estimated, gains = node_model(nodes)
    best = {run: min(costs[n][run] for n in nodes) for run in RUNS}

    def capacity(mix):
        return sum(gains[n] / sum(p * costs[n][r] for r, p in mix.items()) for n in nodes)
    return best, capacity


def parse_slo(text):
    slo = {}
    for part in text.split(','):
        key, _, value = part.partition('=')
        if key.strip() not in ('chat', 'text', 'image', 'audio') or not value:
            raise SystemExit(f'bad --slo entry {part!r}: use <modality>=<factor>')
        slo[key.strip()] = float(value)
    return slo


def make(args):
    mix_spec = MIXES.get(args.mix, args.mix)
    mix = parse_mix(mix_spec)
    best, capacity = best_costs(args.nodes) if args.nodes else (None, None)
    if args.load is not None:
        if capacity is None and args.capacity is None:
            raise SystemExit('--load needs --nodes (to compute capacity) or --capacity')
        cap = args.capacity if args.capacity is not None else capacity(mix)
        rate = args.load * cap
    else:
        rate = args.rate
    if not rate or rate <= 0:
        raise SystemExit('give --rate, or --load with --nodes or --capacity')

    streams = {k: random.Random(f'{args.seed}:{v}') for k, v in SEED_STREAMS.items()}
    if args.arrivals == 'replay':
        if not args.replay:
            raise SystemExit('--arrivals replay needs --replay <csv>')
        times = arrivals_from_replay(args.replay, args.replay_column, args.replay_unit, rate,
                                     args.duration, args.replay_offset)
        segments = None
    else:
        segments = SHAPES[args.arrivals](rate, args.duration, args, streams['bursts'])
        times = arrivals_from_shape(segments, streams['arrivals'])

    slo = parse_slo(args.slo)
    wl, used, rows = streams['workloads'], {}, []
    runs, weights = list(mix), list(mix.values())
    for t, burst in times:
        run = wl.choices(runs, weights=weights)[0]
        modality, size = RUNS[run]
        used[modality] = used.get(modality, 0) + 1   # a new input per request; input 0 = warm-up
        factor = slo[modality]
        rows.append({'id': len(rows) + 1, 'arrival_s': f'{t:.3f}', 'run': run, 'modality': modality,
                     'size': size, 'input': used[modality], 'slo_factor': f'{factor:g}',
                     'deadline_s': f'{factor * best[run]:.1f}' if best else '',
                     'burst': int(burst)})

    name = args.out or TRACES_DIR / 'phase2' / default_name(args, rate)
    out = Path(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != 'func'}
    meta = {'params': params, 'mix': mix_spec, 'rate_req_s': rate,
            'capacity_req_s': (args.capacity if args.capacity is not None
                               else capacity(mix) if capacity else None),
            'segments': [[round(s, 3), round(e, 3), round(r, 5), b] for s, e, r, b in segments]
            if segments and len(segments) <= 2000 else None,
            'stats': stats(rows, args.duration)}
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print_stats(out, meta['stats'])


def default_name(args, rate):
    parts = [args.arrivals, args.mix if args.mix in MIXES else 'custom',
             f'load{args.load:g}' if args.load is not None else f'rate{rate:.3g}',
             f'dur{args.duration:g}', f'seed{args.seed}']
    return '-'.join(parts) + '.csv'


# ---------------------------------------------------------------- describe

def stats(rows, duration=None):
    """Summary of a trace: size, mix, mean rate, burstiness."""
    t = [float(r['arrival_s']) for r in rows]
    if len(t) < 2:
        return {'requests': len(t)}
    span = duration or t[-1]
    gaps = [b - a for a, b in zip([0.0] + t[:-1], t)]
    mean_gap = statistics.fmean(gaps)
    window = 60.0   # peak load over any 60 s window, relative to the mean
    peak = max(bisect.bisect_right(t, x + window) - i for i, x in enumerate(t))
    counts = {m: sum(r['modality'] == m for r in rows) for m in ('chat', 'text', 'image', 'audio')}
    in_burst = sum(int(r.get('burst') or 0) for r in rows)
    return {'requests': len(t), 'span_s': round(span, 1), 'mean_req_s': round(len(t) / span, 4),
            'interarrival_cv': round(statistics.pstdev(gaps) / mean_gap, 2),   # 1 = Poisson, >1 = bursty
            'peak_60s_vs_mean': round(peak / (window * len(t) / span), 2),
            'requests_in_bursts': in_burst, 'per_modality': counts}


def print_stats(path, s):
    print(f'{path}: {s["requests"]} requests', end='')
    if 'mean_req_s' in s:
        mods = ', '.join(f'{m} {n}' for m, n in s['per_modality'].items())
        print(f' over {s["span_s"]} s, mean {s["mean_req_s"]} req/s, interarrival CV '
              f'{s["interarrival_cv"]}, peak 60 s load {s["peak_60s_vs_mean"]}x mean, '
              f'{s["requests_in_bursts"]} in bursts; {mods}')
    else:
        print()


def describe(args):
    for path in args.traces:
        with open(path, newline='', encoding='utf-8') as f:
            print_stats(path, stats(list(csv.DictReader(f))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('make', help='write one trace (CSV + JSON)')
    p.add_argument('--arrivals', choices=[*SHAPES, 'replay'], default='poisson')
    p.add_argument('--rate', type=float, help='mean arrivals per second (base rate for step and ramp)')
    p.add_argument('--load', type=float, help='rate as a fraction of cluster capacity for this mix')
    p.add_argument('--capacity', type=float, help='cluster capacity in req/s (else computed from --nodes)')
    p.add_argument('--nodes', help='nodes.json: capacity for --load, and absolute deadlines')
    p.add_argument('--duration', type=float, default=900, help='seconds of arrivals (default 900)')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--mix', default='default', help=f'one of {", ".join(MIXES)}, or chat=..,text=.. / run ids')
    p.add_argument('--slo', default=SLO_DEFAULT, help=f'deadline factor per modality (default {SLO_DEFAULT})')
    g = p.add_argument_group('onoff (bursty)')
    g.add_argument('--burst-factor', type=float, default=3.0, help='rate in a burst / mean rate (default 3)')
    g.add_argument('--burst-fraction', type=float, default=0.2, help='share of time in bursts (default 0.2)')
    g.add_argument('--burst-length', type=float, default=60.0, help='mean burst length, s (default 60)')
    g = p.add_argument_group('step (flash crowd)')
    g.add_argument('--step-at', type=float, default=300.0)
    g.add_argument('--step-length', type=float, default=120.0)
    g.add_argument('--step-factor', type=float, default=3.0)
    g = p.add_argument_group('ramp')
    g.add_argument('--ramp-from', type=float, default=0.3, help='start, as a multiple of --rate')
    g.add_argument('--ramp-to', type=float, default=1.2, help='end, as a multiple of --rate')
    g = p.add_argument_group('replay (arrival times from a real trace)')
    g.add_argument('--replay', help='CSV with one arrival time per row')
    g.add_argument('--replay-column', default='timestamp')
    g.add_argument('--replay-unit', choices=['s', 'ms', 'us'], default='s')
    g.add_argument('--replay-offset', type=int, default=0, help='first arrival to use (pick another window)')
    p.add_argument('--out', help='trace file (default traces/phase2/<arrivals>-<mix>-<load>-<dur>-<seed>.csv)')
    p.set_defaults(func=make)
    p = sub.add_parser('describe', help='summary statistics of existing traces')
    p.add_argument('traces', nargs='+')
    p.set_defaults(func=describe)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
