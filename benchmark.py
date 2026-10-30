#!/usr/bin/env python3
"""Phase 1 single-node benchmark: HTTP inference + CPU/RSS sampling of llama-server.

Run ON the inference node (in tmux), next to llama-server, so network and laptop
upload time stay out of the latency numbers. Runs alongside dolly.py, coco.py,
cnn_dailymail.py and librispeech.py.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
import math
import shlex
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

from coco import COCODataset
from cnn_dailymail import CNNDailyMailDataset
from dolly import DollyQADataset
from librispeech import LibriSpeechDataset

REPEATS = 5          # measured requests per workload (each a different input), times --concurrency
WARMUPS = 1
OUTPUT_TOKENS = 128
TIMEOUT_SECONDS = 900

RUNS = {
    "00": ("chat", "short"),
    "01": ("text", 512), "02": ("text", 1024), "03": ("text", 3072),
    "04": ("image", 1), "05": ("image", 2), "06": ("image", 4),
    "07": ("audio", 10), "08": ("audio", 40),   # one vs two 30 s audio windows, see librispeech.py
}
FIELDS = ["timestamp_utc", "model", "llama_build", "node", "run", "modality", "size", "size_unit",
          "concurrency", "requests", "success", "failed", "median_e2e_s", "p95_s", "ttft_s",
          "prompt_tokens", "output_tokens", "req_s", "out_tok_s", "decode_tok_s",
          "peak_ram_gb", "cpu_pct"]
REQUEST_FIELDS = ["timestamp_utc", "model", "node", "run", "modality", "size", "concurrency",
                  "index", "input", "sent_s", "status", "error", "e2e_s", "ttft_s",
                  "prompt_tokens", "output_tokens", "decode_tok_s",
                  "server_prompt_ms", "server_predicted_ms"]
SIZE_UNITS = {"chat": "question", "text": "tokens", "image": "images", "audio": "seconds"}
RESULTS_DIR = Path("results/phase1")   # one folder per node + model, so teammates' results never collide

# Samples llama-server's CPU and RSS; runs locally, or remotely over SSH with --ssh.
# Uses only Python's standard library.
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


def payload(ds, modality, setting, index):
    return {'messages': messages_for(ds, modality, setting, index),
            'max_tokens': OUTPUT_TOKENS, 'temperature': 0,
            'stream': True, 'stream_options': {'include_usage': True},
            'cache_prompt': False,
            'chat_template_kwargs': {'enable_thinking': False}}  # same request shape as roundrobin.py


class RemoteSamples:
    """Samples llama-server for the duration of a `with` block: locally when ssh_target
    is None, otherwise through one SSH process (no extra resident program on EC2)."""
    def __init__(self, ssh_target):
        self.ssh_target = ssh_target
        self.samples = []
        self.problem = None
        self.proc = None
        self.reader = None

    def __enter__(self):
        if self.ssh_target is None:
            command = [sys.executable, "-u", "-c", REMOTE_SAMPLER]
        else:
            command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                       self.ssh_target, "python3 -u -c " + shlex.quote(REMOTE_SAMPLER)]
        self.proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True)
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
        # Allow the sampler to start before timing HTTP inference.
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
        # prompt_tokens shows how many tokens an image or audio clip became.
        return {"status":"ok", "error":"", "e2e_s":elapsed,
                "ttft_s":first, "decode_tok_s":rate, "output_tokens":n,
                "prompt_tokens":usage.get('prompt_tokens') or timing.get('prompt_n'),
                "server_prompt_ms":timing.get('prompt_ms'),
                "server_predicted_ms":timing.get('predicted_ms')}
    except Exception as exc:
        return {"status":"error", "error":str(exc)[:180],
                "e2e_s":time.perf_counter()-start, "ttft_s":None,
                "decode_tok_s":None, "output_tokens":None, "prompt_tokens":None,
                "server_prompt_ms":None, "server_predicted_ms":None}


def instance_type():
    """EC2 instance type from the instance metadata service (IMDSv2); hostname elsewhere."""
    try:
        token = requests.put('http://169.254.169.254/latest/api/token', timeout=1,
                             headers={'X-aws-ec2-metadata-token-ttl-seconds': '60'}).text
        return requests.get('http://169.254.169.254/latest/meta-data/instance-type', timeout=1,
                            headers={'X-aws-ec2-metadata-token': token}).text.strip()
    except requests.RequestException:
        return socket.gethostname()


def server_info(url):
    """Model file actually served (e.g. Qwen2.5-Omni-3B-Q8_0) and llama.cpp build."""
    url = url.rstrip('/')
    model = Path(requests.get(url + '/v1/models', timeout=10).json()['data'][0]['id']).stem
    try:
        build = requests.get(url + '/props', timeout=10).json().get('build_info', '')
    except (requests.RequestException, ValueError):
        build = ''
    return model, build


def append_rows(path, fields, rows):
    """Add result rows, without overwriting earlier benchmark runs."""
    path.parent.mkdir(exist_ok=True, parents=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    if not needs_header:
        with path.open('r', newline='', encoding='utf-8') as f:
            existing_header = next(csv.reader(f), [])
        if existing_header != fields:
            raise RuntimeError(
                f"{path} has a different column format. "
                "Rename or back up the old file before running this version."
            )
    with path.open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Phase 1 single-node benchmark.')
    parser.add_argument('--run', choices=RUNS.keys(), help='One run 00–08; default: all nine')
    parser.add_argument('--repeats', type=int, default=REPEATS,
                        help='measured requests per workload (times --concurrency)')
    parser.add_argument('--concurrency', type=int, default=1,
                        help='requests in flight at once; keep <= llama-server slots (-np)')
    parser.add_argument('--node', help='node label in the results; default: this EC2 instance type '
                                       '(add a suffix for variants, e.g. c7i.2xlarge-noamx)')
    parser.add_argument('--url', default='http://127.0.0.1:8080', help='llama-server address')
    parser.add_argument('--ssh', help='sample CPU/RAM on this SSH host instead of locally '
                                      '(only when the client is not on the node)')
    args = parser.parse_args()
    selected = {args.run: RUNS[args.run]} if args.run else RUNS
    node = args.node or instance_type()
    model, build = server_info(args.url)
    out_dir = RESULTS_DIR / f"{node}__{model}"
    print(f'node={node} model={model} build={build} concurrency={args.concurrency} -> {out_dir}', flush=True)

    datasets = {}
    for modality, _ in selected.values():
        if modality not in datasets:
            if modality == 'chat':
                datasets[modality] = DollyQADataset()
            elif modality == 'text':
                datasets[modality] = CNNDailyMailDataset()
            elif modality == 'image':
                datasets[modality] = COCODataset()
            else:
                datasets[modality] = LibriSpeechDataset()

    n = args.repeats * args.concurrency
    with requests.Session() as session:
        for run, (modality, setting) in selected.items():
            print(f'[{node}] {run} {modality}={setting} x{n}, {args.concurrency} at a time', flush=True)
            ds = datasets[modality]
            for w in range(WARMUPS):
                record = request_once(session, args.url, payload(ds, modality, setting, w))
                print(f"  warmup: {record['status']} " + (record['error'] or f"{record['e2e_s']:.2f}s"), flush=True)
            # Build payloads outside the measured interval; inputs after the warm-up ones.
            payloads = [payload(ds, modality, setting, WARMUPS + k) for k in range(n)]

            def send(k):
                sent = time.perf_counter() - start
                with requests.Session() as s:
                    record = request_once(s, args.url, payloads[k])
                print(f"  {k+1}: {record['status']} " + (record['error'] or f"{record['e2e_s']:.2f}s"), flush=True)
                return {'index': k + 1, 'input': WARMUPS + k, 'sent_s': sent, **record}

            with RemoteSamples(args.ssh) as monitor:
                start = time.perf_counter()
                # Closed loop: the pool keeps `concurrency` requests outstanding until all n are sent.
                with ThreadPoolExecutor(args.concurrency) as pool:
                    rows = list(pool.map(send, range(n)))
                makespan = time.perf_counter() - start
            if monitor.problem:
                print('  CPU/RAM monitoring:', monitor.problem)
            cpu, ram = monitor.metrics()

            stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
            ok = [r for r in rows if r['status'] == 'ok']
            valid = lambda field: [r[field] for r in ok if r[field] is not None]
            summary = {
                'timestamp_utc':stamp, 'model':model, 'llama_build':build, 'node':node,
                'run':run, 'modality':modality, 'size':setting, 'size_unit':SIZE_UNITS[modality],
                'concurrency':args.concurrency, 'requests':n,
                'success':len(ok), 'failed':len(rows)-len(ok),
                'median_e2e_s':median(valid('e2e_s')),
                'p95_s':percentile95(valid('e2e_s')),
                'ttft_s':median(valid('ttft_s')),
                'prompt_tokens':median(valid('prompt_tokens')),
                'output_tokens':median(valid('output_tokens')),
                'req_s':len(ok)/makespan,
                'out_tok_s':sum(valid('output_tokens'))/makespan,
                'decode_tok_s':median(valid('decode_tok_s')),
                'peak_ram_gb':ram, 'cpu_pct':cpu,
            }
            append_rows(out_dir / 'summary.csv', FIELDS, [summary])
            append_rows(out_dir / 'requests.csv', REQUEST_FIELDS,
                        [{'timestamp_utc':stamp, 'model':model, 'node':node, 'run':run,
                          'modality':modality, 'size':setting, 'concurrency':args.concurrency, **r}
                         for r in rows])
            print('  result:', summary, flush=True)
    print(f'Appended results to {out_dir}/summary.csv and requests.csv')

if __name__ == '__main__':
    main()
