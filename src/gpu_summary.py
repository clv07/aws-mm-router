#!/usr/bin/env python3
"""GPU utilization, memory, power and energy per request for each benchmark pass on a GPU node.

gpu_experiment.sh samples nvidia-smi every 0.5 s into <results folder>/logs/gpu-<run_id>.csv.
This joins those samples with requests_v2.csv: a pass (one workload at one slot count,
concurrency and output mode) runs from its first request's send time to its last request's
finish time, and every GPU sample in that window belongs to it. Writes
<results folder>/gpu_summary.csv, rebuilt from all gpu-*.csv logs each time.

    python src/gpu_summary.py results/phase1/g4dn.xlarge-gpu__Qwen2.5-Omni-3B-Q8_0

requests_v2.csv stores finish times to the second, so each window is accurate to about 1 s;
short passes (a few seconds) have few samples, and their numbers are rough.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import statistics

FIELDS = ["run_id", "node", "model", "slots", "concurrency", "output_mode", "run", "modality",
          "size", "requests", "ok", "window_s", "req_s", "out_tok_s", "median_e2e_s",
          "gpu_util_median_pct", "gpu_util_mean_pct", "mem_bw_util_median_pct",
          "vram_peak_gb", "power_mean_w", "energy_j_per_request", "sm_clock_median_mhz",
          "temp_max_c", "samples"]


def read_gpu_log(path):
    """nvidia-smi --format=csv,nounits rows -> list of (utc datetime, dict of floats)."""
    samples = []
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = [h.strip().split(' [')[0] for h in next(reader, [])]
        for row in reader:
            if len(row) != len(header):
                continue
            item = dict(zip(header, (c.strip() for c in row)))
            try:
                ts = datetime.strptime(item['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc)
            except (KeyError, ValueError):
                continue
            values = {}
            for key in ('utilization.gpu', 'utilization.memory', 'memory.used', 'power.draw',
                        'temperature.gpu', 'clocks.sm'):
                try:
                    values[key] = float(item[key])
                except (KeyError, ValueError):   # "[N/A]" on some GPUs
                    values[key] = None
            samples.append((ts, values))
    return samples


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize(folder):
    folder = Path(folder)
    logs = {p.stem.removeprefix('gpu-'): p for p in sorted((folder / 'logs').glob('gpu-*.csv'))}
    if not logs:
        raise SystemExit(f'no logs/gpu-<run_id>.csv in {folder}')
    passes = defaultdict(list)
    with (folder / 'requests_v2.csv').open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['run_id'] in logs:
                key = (row['run_id'], row['slots'], row['concurrency'], row['output_mode'], row['run'])
                passes[key].append(row)

    out = []
    gpu = {run_id: read_gpu_log(path) for run_id, path in logs.items()}
    for (run_id, slots, conc, mode, run), rows in sorted(
            passes.items(), key=lambda kv: (kv[0][0], int(kv[0][1] or 0), int(kv[0][2]), kv[0][3], kv[0][4])):
        finish = [datetime.fromisoformat(r['timestamp_utc']) for r in rows]
        e2e = [number(r['e2e_s']) or 0.0 for r in rows]
        # Finish times are rounded down to the second: widen the end by 1 s.
        start = min(t - timedelta(seconds=d) for t, d in zip(finish, e2e))
        end = max(finish) + timedelta(seconds=1)
        window = (end - start).total_seconds()
        inside = [v for ts, v in gpu[run_id] if start <= ts <= end]
        col = lambda k: [v[k] for v in inside if v[k] is not None]
        ok = [r for r in rows if r['status'] == 'ok']
        out_tokens = sum(number(r['output_tokens']) or 0 for r in ok)
        power = col('power.draw')
        out.append({
            'run_id': run_id, 'node': rows[0]['node'], 'model': rows[0]['model'], 'slots': slots,
            'concurrency': conc, 'output_mode': mode, 'run': run, 'modality': rows[0]['modality'],
            'size': rows[0]['size'], 'requests': len(rows), 'ok': len(ok),
            'window_s': round(window, 2),
            'req_s': round(len(ok) / window, 4) if window else None,
            'out_tok_s': round(out_tokens / window, 2) if window else None,
            'median_e2e_s': round(statistics.median([number(r['e2e_s']) for r in ok]), 3) if ok else None,
            'gpu_util_median_pct': statistics.median(col('utilization.gpu')) if col('utilization.gpu') else None,
            'gpu_util_mean_pct': round(statistics.mean(col('utilization.gpu')), 1) if col('utilization.gpu') else None,
            'mem_bw_util_median_pct': statistics.median(col('utilization.memory')) if col('utilization.memory') else None,
            'vram_peak_gb': round(max(col('memory.used')) / 1024, 2) if col('memory.used') else None,
            'power_mean_w': round(statistics.mean(power), 1) if power else None,
            # mean power x pass duration, shared by the requests that finished in it
            'energy_j_per_request': round(statistics.mean(power) * window / len(ok), 1) if power and ok else None,
            'sm_clock_median_mhz': statistics.median(col('clocks.sm')) if col('clocks.sm') else None,
            'temp_max_c': max(col('temperature.gpu')) if col('temperature.gpu') else None,
            'samples': len(inside),
        })
    path = folder / 'gpu_summary.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out)
    print(f'wrote {len(out)} passes to {path}')
    for r in out:
        if r['output_mode'] == 'fixed':
            print(f"  slots={r['slots']:>2} c={r['concurrency']:>2} {r['run']} {r['modality']:<5} {r['size']:>5}: "
                  f"{r['req_s']} req/s, {r['out_tok_s']} out tok/s, GPU {r['gpu_util_median_pct']}%, "
                  f"{r['power_mean_w']} W, {r['energy_j_per_request']} J/request")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('folder', help='results/phase1/<node>__<model> folder of a GPU run')
    summarize(parser.parse_args().folder)
