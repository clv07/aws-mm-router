#!/usr/bin/env python3
"""One-client benchmark: remote HTTP inference + remote CPU/RSS sampling via SSH.

Run alongside coco.py, cnn_dailymail.py and librispeech.py on the CLIENT.
Configure NODES below; SSH access is via your normal ~/.ssh/config or ssh-agent.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import shlex
import statistics
import subprocess
import threading
import time
from pathlib import Path

import requests

from coco import COCODataset
from cnn_dailymail import CNNDailyMailDataset
from librispeech import LibriSpeechDataset

# Configure model and nodes here; URLs must be reachable from the benchmark client.
MODEL = "Qwen2.5-Omni-3B"

NODES = {
    "c8g.4xlarge ": {
        "url": "http://127.0.0.1:8080",
        "ssh": "qwen-large"
    }
}

REPEATS = 1
WARMUPS = 1
OUTPUT_TOKENS = 128
TIMEOUT_SECONDS = 900

RUNS = {
    "01": ("text", 512), "02": ("text", 1024), "03": ("text", 3072),
    "04": ("image", 1), "05": ("image", 2), "06": ("image", 4),
    "07": ("audio", 5), "08": ("audio", 10), "09": ("audio", 20),
}
FIELDS = ["timestamp_utc", "model", "node", "run", "modality", "size", "size_unit",
          "success", "failed", "median_e2e_s", "p95_s", "req_s", "ttft_s",
          "decode_tok_s", "peak_ram_gb", "cpu_pct"]
SIZE_UNITS = {"text": "tokens", "image": "images", "audio": "seconds"}
OUTPUT_CSV = Path("results/benchmark.csv")

# Executed on remote instances over SSH; uses only Python's standard library.
REMOTE_SAMPLER = r'''
import json, os, time
pids = []
for x in os.listdir('/proc'):
    if not x.isdigit():
        continue
    try:
        name = open('/proc/'+x+'/comm').read().strip()
        if name.startswith('llama-server') or name.startswith('llama-serv'):
            pids.append(int(x))
    except (OSError, ValueError):
        pass
if not pids:
    print(json.dumps({'error':'llama-server process not found'}), flush=True)
    raise SystemExit(1)
pid = max(pids)
cores = os.cpu_count() or 1
previous = None
while True:
    try:
        stat = open(f'/proc/{pid}/stat').read()
        after_comm = stat[stat.rfind(')') + 2:].split()
        cpu_ticks = int(after_comm[11]) + int(after_comm[12])
        rss_kib = None
        for line in open(f'/proc/{pid}/status'):
            if line.startswith('VmRSS:'):
                rss_kib = int(line.split()[1]); break
        now = time.monotonic()
        cpu = None
        if previous is not None:
            elapsed = now - previous[0]
            if elapsed > 0:
                cpu = 100.0 * (cpu_ticks - previous[1]) / os.sysconf('SC_CLK_TCK') / elapsed / cores
        print(json.dumps({'cpu_pct':cpu, 'ram_gb':rss_kib / (1024**2) if rss_kib is not None else None}), flush=True)
        previous = (now, cpu_ticks)
        time.sleep(0.2)
    except (OSError, ValueError):
        break
'''


def percentile95(values):
    xs = sorted(values)
    if not xs:
        return None
    pos = (len(xs) - 1) * .95
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def median(values):
    return statistics.median(values) if values else None


def messages_for(ds, modality, setting, index):
    original = ds.messages(setting, start=index)
    result = []
    for message in original:
        parts = message["content"]
        if isinstance(parts, str):
            result.append({"role": message["role"], "content": parts})
            continue
        output = []
        for item in parts:
            if item["type"] == "text":
                output.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image":
                output.append({"type": "image_url", "image_url": {"url": item["image"]}})
            elif item["type"] == "audio":
                # The uploaded LibriSpeech script produces a WAV data URL.
                output.append({"type": "input_audio", "input_audio": {"data": item["audio"]}})
            else:
                raise ValueError(f'Unsupported message type: {item["type"]}')
        result.append({"role": message["role"], "content": output})
    return result


class RemoteSamples:
    """One SSH process per request; no extra resident program on EC2."""
    def __init__(self, ssh_target):
        self.ssh_target = ssh_target
        self.samples = []
        self.problem = None
        self.proc = None
        self.reader = None

    def __enter__(self):
        self.proc = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             self.ssh_target, "python3 -u -c " + shlex.quote(REMOTE_SAMPLER)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        def read():
            for line in self.proc.stdout:
                try:
                    item = json.loads(line)
                    if "error" in item:
                        self.problem = item["error"]
                    else:
                        self.samples.append(item)
                except json.JSONDecodeError:
                    continue
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        # Allow the remote process to start before timing HTTP inference.
        # This wait is outside the request latency measurement.
        until = time.monotonic() + 5
        while not self.samples and not self.problem and self.proc.poll() is None and time.monotonic() < until:
            time.sleep(.05)
        return self

    def __exit__(self, *_):
        if self.proc.poll() is None:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.reader.join(timeout=2)

    def metrics(self):
        cpu = [s["cpu_pct"] for s in self.samples if s.get("cpu_pct") is not None]
        ram = [s["ram_gb"] for s in self.samples if s.get("ram_gb") is not None]
        return median(cpu) if cpu else None, max(ram) if ram else None


def request_once(session, url, payload):
    start = time.perf_counter()
    first = None
    usage, timing = {}, {}
    count = 0
    try:
        with session.post(url.rstrip('/') + '/v1/chat/completions', json=payload,
                          stream=True, timeout=(15, TIMEOUT_SECONDS)) as response:
            response.raise_for_status()
            for raw in response.iter_lines():
                if not raw or not raw.startswith(b'data:'):
                    continue
                data = raw[5:].strip()
                if data == b'[DONE]':
                    break
                event = json.loads(data)
                if event.get('error'):
                    raise RuntimeError(str(event['error']))
                usage.update(event.get('usage') or {})
                timing.update(event.get('timings') or {})
                for choice in event.get('choices', []):
                    content = (choice.get('delta') or {}).get('content')
                    if isinstance(content, str) and content:
                        if first is None:
                            first = time.perf_counter() - start
                        count += 1
        elapsed = time.perf_counter() - start
        if first is None:
            raise RuntimeError('No output text token returned in stream')
        n = usage.get('completion_tokens') or timing.get('predicted_n')
        rate = timing.get('predicted_per_second')
        if rate is None and n and timing.get('predicted_ms'):
            rate = n * 1000 / timing['predicted_ms']
        # Client-side estimate when streaming response lacks server timing.
        if rate is None and n and elapsed > first:
            rate = max(0, n - 1) / (elapsed - first)
        return {"status":"ok", "error":"", "e2e_s":elapsed,
                "ttft_s":first, "decode_tok_s":rate, "output_tokens":n}
    except Exception as exc:
        return {"status":"error", "error":str(exc)[:180],
                "e2e_s":time.perf_counter()-start, "ttft_s":None,
                "decode_tok_s":None, "output_tokens":None}


def append_summary(row):
    """Add one result row, without overwriting earlier benchmark runs."""
    OUTPUT_CSV.parent.mkdir(exist_ok=True, parents=True)
    needs_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
    if not needs_header:
        with OUTPUT_CSV.open('r', newline='', encoding='utf-8') as f:
            existing_header = next(csv.reader(f), [])
        if existing_header != FIELDS:
            raise RuntimeError(
                f"{OUTPUT_CSV} has a different column format. "
                "Rename or back up the old file before running this version."
            )
    with OUTPUT_CSV.open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Remote Qwen Omni benchmark (one client script).')
    parser.add_argument('--run', choices=RUNS.keys(), help='One run 01–09; default: all nine')
    args = parser.parse_args()
    selected = {args.run: RUNS[args.run]} if args.run else RUNS
    datasets = {}
    for modality, _ in selected.values():
        if modality not in datasets:
            if modality == 'text':
                datasets[modality] = CNNDailyMailDataset()
            elif modality == 'image':
                datasets[modality] = COCODataset()
            else:
                datasets[modality] = LibriSpeechDataset()
    with requests.Session() as session:
        for node, config in NODES.items():
            for run, (modality, setting) in selected.items():
                print(f'[{node}] {run} {modality}={setting}', flush=True)
                rows = []
                for i in range(-WARMUPS, REPEATS):
                    # Build payload outside the measured interval.
                    payload = {
                        'messages': messages_for(datasets[modality], modality, setting, max(0, i + WARMUPS)),
                        'max_tokens': OUTPUT_TOKENS, 'temperature': 0,
                        'stream': True, 'stream_options': {'include_usage': True},
                        'cache_prompt': False,
                    }
                    with RemoteSamples(config['ssh']) as monitor:
                        record = request_once(session, config['url'], payload)
                    record['cpu_pct'], record['peak_ram_gb'] = monitor.metrics()
                    if monitor.problem:
                        print('  SSH monitoring:', monitor.problem)
                    if i >= 0:
                        rows.append({'node':node, 'run':run, 'index':i+1, **record})
                    print(f"  {'warmup' if i<0 else i+1}: {record['status']} "+
                          (record['error'] or f"{record['e2e_s']:.2f}s"), flush=True)
                wall_s = sum(r["e2e_s"] for r in rows)
                ok = [r for r in rows if r['status']=='ok']
                valid = lambda field: [r[field] for r in ok if r[field] is not None]
                summary = {
                    'timestamp_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    'model':MODEL, 'node':node, 'run':run, 'modality':modality,
                    'size':setting, 'size_unit':SIZE_UNITS[modality],
                    'success':len(ok), 'failed':len(rows)-len(ok),
                    'median_e2e_s':median(valid('e2e_s')),
                    'p95_s':percentile95(valid('e2e_s')),
                    'req_s':len(ok)/wall_s if wall_s else None,
                    'ttft_s':median(valid('ttft_s')),
                    'decode_tok_s':median(valid('decode_tok_s')),
                    'peak_ram_gb':max(valid('peak_ram_gb'), default=None),
                    'cpu_pct':median(valid('cpu_pct')),
                }
                append_summary(summary)
                print('  result:', summary, flush=True)
    print(f'Appended results to {OUTPUT_CSV}')

if __name__ == '__main__':
    main()
