#!/usr/bin/env python3
"""Round-robin baseline over the NODES below.

A shuffled text/image/audio mix is sent in list order with at most --inflight requests
outstanding across all nodes; request i goes to node i % len(NODES).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
from pathlib import Path
import random
import time

import requests

from benchmark import RemoteSamples, median, messages_for, percentile95, request_once
from coco import COCODataset
from cnn_dailymail import CNNDailyMailDataset
from librispeech import LibriSpeechDataset

# Nodes to round-robin over, in turn order (first = node A). One local port per node's SSH tunnel.
NODES = {
    "c7a.2xlarge": {
        "url": "http://127.0.0.1:8080",
        "ssh": "node-c7a"
    },
    "c7i.2xlarge": {
        "url": "http://127.0.0.1:8081",
        "ssh": "node-c7i"
    },
}

OUTPUT_TOKENS = 128
PATTERN = ["text", "text", "text", "image", "audio"]  # mix ratio 3:1:1
SIZES = {"text": [512, 1024, 3072], "image": [1, 2, 4], "audio": [5, 10, 20]}  # tokens / images / seconds
FIELDS = ["timestamp_utc", "model", "policy", "requests", "inflight", "seed", "group", "success", "failed",
          "median_e2e_s", "p95_s", "ttft_s", "req_s", "peak_ram_gb", "cpu_pct"]
OUTPUT_CSV = Path("results/roundrobin.csv")
REQUESTS_CSV = Path("results/roundrobin_requests.csv")  # one row per request, all runs


def payload(ds, modality, size, index):
    return {'messages': messages_for(ds, modality, size, index),
            'max_tokens': OUTPUT_TOKENS, 'temperature': 0,
            'stream': True, 'stream_options': {'include_usage': True},
            'cache_prompt': False,
            'chat_template_kwargs': {'enable_thinking': False}}  # Gemma 4 otherwise spends all tokens "thinking"


def main():
    parser = argparse.ArgumentParser(description='Round-robin baseline over NODES.')
    parser.add_argument('--requests', type=int, default=50, help='total requests')
    parser.add_argument('--inflight', type=int, default=4, help='max requests outstanding across all nodes')
    parser.add_argument('--seed', type=int, default=0, help='request order')
    args = parser.parse_args()
    datasets = {'text': CNNDailyMailDataset(), 'image': COCODataset(), 'audio': LibriSpeechDataset()}
    names = list(NODES)
    # Model each node is actually serving, e.g. "Qwen2.5-Omni-3B-Q8_0", as reported by llama-server.
    models = {n: Path(requests.get(c['url'].rstrip('/') + '/v1/models', timeout=10).json()['data'][0]['id']).stem
              for n, c in NODES.items()}
    run_model = '+'.join(sorted(set(models.values())))  # one name unless the nodes serve different models

    # Workload: 3:1:1 text/image/audio, sizes cycle within each modality, order shuffled.
    jobs = []
    for k in range(args.requests):
        m = PATTERN[k % len(PATTERN)]
        jobs.append((m, SIZES[m][k // len(PATTERN) % len(SIZES[m])], k))
    random.Random(args.seed).shuffle(jobs)
    # Build payloads outside the measured interval.
    payloads = [payload(datasets[m], m, size, i) for m, size, i in jobs]

    print('warmup', flush=True)
    for config in NODES.values():
        for m in SIZES:
            request_once(requests.Session(), config['url'], payload(datasets[m], m, SIZES[m][0], 0))

    def send(i):
        node = names[i % len(names)]  # round robin
        sent = time.perf_counter() - start
        record = request_once(requests.Session(), NODES[node]['url'], payloads[i])
        print(f"  {i+1:>3} {node} {jobs[i][0]}={jobs[i][1]}: {record['status']} "
              + (record['error'] or f"{record['e2e_s']:.2f}s"), flush=True)
        return {'index': i+1, 'node': node, 'model': models[node], 'modality': jobs[i][0], 'size': jobs[i][1],
                'sent_s': sent, **record}

    with ExitStack() as stack:
        monitors = {n: stack.enter_context(RemoteSamples(c['ssh'])) for n, c in NODES.items()}
        print(f'sending {len(jobs)} requests to {names}, {args.inflight} in flight', flush=True)
        start = time.perf_counter()
        # The pool starts requests in list order; each time one finishes, the next one is sent.
        with ThreadPoolExecutor(args.inflight) as pool:
            rows = list(pool.map(send, range(len(jobs))))
        makespan = time.perf_counter() - start

    # Per-request results, appended to one file; the first columns identify the run.
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    run = {'timestamp_utc':stamp, 'policy':'round_robin', 'seed':args.seed, 'inflight':args.inflight}
    REQUESTS_CSV.parent.mkdir(exist_ok=True, parents=True)
    needs_header = not REQUESTS_CSV.exists()
    with REQUESTS_CSV.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[*run, *rows[0]])
        if needs_header:
            writer.writeheader()
        writer.writerows({**run, **r} for r in rows)

    # Summary rows: all requests, each modality, each node.
    groups = {'all': rows,
              **{m: [r for r in rows if r['modality'] == m] for m in SIZES},
              **{n: [r for r in rows if r['node'] == n] for n in names}}
    needs_header = not OUTPUT_CSV.exists()
    with OUTPUT_CSV.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if needs_header:
            writer.writeheader()
        for group, members in groups.items():
            ok = [r for r in members if r['status'] == 'ok']
            valid = lambda field: [r[field] for r in ok if r[field] is not None]
            cpu, ram = monitors[group].metrics() if group in monitors else (None, None)
            summary = {
                'timestamp_utc':stamp, 'model':models.get(group, run_model), 'policy':'round_robin',
                'requests':args.requests, 'inflight':args.inflight,
                'seed':args.seed, 'group':group,
                'success':len(ok), 'failed':len(members)-len(ok),
                'median_e2e_s':median(valid('e2e_s')),
                'p95_s':percentile95(valid('e2e_s')),
                'ttft_s':median(valid('ttft_s')),
                'req_s':len(ok)/makespan,
                'peak_ram_gb':ram, 'cpu_pct':cpu,
            }
            writer.writerow(summary)
            print('  result:', summary, flush=True)
    print(f'Appended results to {REQUESTS_CSV} and {OUTPUT_CSV}')

if __name__ == '__main__':
    main()
