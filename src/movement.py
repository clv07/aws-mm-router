#!/usr/bin/env python3
"""Data movement benchmarks: network time vs size along the request path, and network byte counters.
The KV cache moved between nodes is measured by split.py.

  python src/movement.py sink                                     # on a node: endpoint that takes or sends N bytes
  python src/movement.py payload --url http://127.0.0.1:8090 --label router-m8g
  python src/movement.py netdev --hosts node-c7a node-c7i node-m8g --tag before-run

payload: uploads and downloads 1 KB - 128 MB to a `sink` on the other machine, through the same path
the requests take (e.g. the router's SSH tunnel to a node), and fits time = round trip + bytes / bandwidth.

netdev: network byte counters of each host (/proc/net/dev), for total traffic per run (before/after).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import requests

from benchmark import append_rows, median

OUT_DIR = Path("results/movement")
PAYLOAD_SIZES = [1_000, 10_000, 100_000, 1_000_000, 4_000_000, 16_000_000, 64_000_000, 128_000_000]
PAYLOAD_FIELDS = ["run_id", "timestamp_utc", "label", "url", "direction", "bytes", "repeat", "seconds", "mb_s"]
NETDEV_FIELDS = ["timestamp_utc", "tag", "host", "interface", "rx_bytes", "tx_bytes"]
CHUNK = 1 << 20


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


# ---------------------------------------------------------------- sink and payload sweep

class Sink(BaseHTTPRequestHandler):
    """POST: reads and discards the body, answers with its size. GET /bytes/<n>: sends n bytes."""
    protocol_version = 'HTTP/1.1'
    disable_nagle_algorithm = True   # TCP_NODELAY: else small replies wait ~40 ms for a delayed ACK
    filler = os.urandom(CHUNK)   # random, so nothing on the path can compress it

    def log_message(self, *_):
        pass

    def _reply(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        left = int(self.headers.get('Content-Length', 0))
        while left > 0:
            left -= len(self.rfile.read(min(CHUNK, left)))
        self._reply({'bytes': int(self.headers.get('Content-Length', 0))})

    def do_GET(self):
        parts = self.path.strip('/').split('/')
        if len(parts) != 2 or parts[0] != 'bytes' or not parts[1].isdigit():
            self.send_error(404)
            return
        n = int(parts[1])
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', str(n))
        self.end_headers()
        while n > 0:
            self.wfile.write(self.filler[:min(CHUNK, n)])
            n -= min(CHUNK, n)


def sink(args):
    server = ThreadingHTTPServer((args.host, args.port), Sink)
    print(f'sink listening on {args.host}:{args.port} (Ctrl-C to stop)', flush=True)
    server.serve_forever()


def fit(points):
    """Least-squares time = a + bytes / bandwidth over (bytes, seconds); returns (a in s, MB/s)."""
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0
    return my - slope * mx, (1 / slope / 1e6) if slope > 0 else None


def payload_sweep(args):
    url = args.url.rstrip('/')
    run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = OUT_DIR / args.label / 'payload.csv'
    print(f'run_id={run_id} {url} -> {out}', flush=True)
    rows = []
    with requests.Session() as s:
        s.get(f'{url}/bytes/1000', timeout=30).raise_for_status()   # warm-up: open the connection
        for size in args.sizes:
            body = os.urandom(size)
            for direction in ('up', 'down'):
                times = []
                for k in range(args.repeats):
                    t = time.perf_counter()
                    if direction == 'up':
                        r = s.post(url + '/', data=body, timeout=300,
                                   headers={'Content-Type': 'application/octet-stream'})
                        r.raise_for_status()
                    else:
                        with s.get(f'{url}/bytes/{size}', timeout=300, stream=True) as r:
                            r.raise_for_status()
                            got = sum(len(c) for c in r.iter_content(CHUNK))
                        if got != size:
                            raise SystemExit(f'got {got} of {size} bytes')
                    sec = time.perf_counter() - t
                    times.append(sec)
                    rows.append({'run_id': run_id, 'timestamp_utc': now_utc(), 'label': args.label,
                                 'url': url, 'direction': direction, 'bytes': size, 'repeat': k + 1,
                                 'seconds': sec, 'mb_s': size / sec / 1e6})
                print(f'  {direction:>4} {size:>11,} B  median {median(times) * 1000:9.2f} ms  '
                      f'{size / median(times) / 1e6:8.1f} MB/s', flush=True)
    append_rows(out, PAYLOAD_FIELDS, rows)
    for direction in ('up', 'down'):
        rtt, bw = fit([(r['bytes'], r['seconds']) for r in rows if r['direction'] == direction])
        print(f'{direction}: time ≈ {rtt * 1000:.2f} ms + bytes / {bw:.0f} MB/s' if bw else
              f'{direction}: no bandwidth fit (times did not grow with size)')
    print(f'appended {len(rows)} rows to {out}')


# ---------------------------------------------------------------- network counters

def netdev(args):
    """Received/sent bytes per network interface (loopback left out), locally or over SSH."""
    rows = []
    for host in args.hosts or ['local']:
        if host == 'local':
            text = Path('/proc/net/dev').read_text()
        else:
            text = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, 'cat /proc/net/dev'],
                                  check=True, capture_output=True, text=True).stdout
        for line in text.splitlines()[2:]:
            name, _, fields = line.partition(':')
            name, fields = name.strip(), fields.split()
            if name != 'lo' and fields:
                rows.append({'timestamp_utc': now_utc(), 'tag': args.tag, 'host': host,
                             'interface': name, 'rx_bytes': int(fields[0]), 'tx_bytes': int(fields[8])})
                print(f'{host:<12} {name:<10} rx {int(fields[0]):>16,}  tx {int(fields[8]):>16,}')
    if args.out:
        append_rows(Path(args.out), NETDEV_FIELDS, rows)
        print(f'appended to {args.out}')


def main():
    parser = argparse.ArgumentParser(description='Data movement benchmarks.')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('sink', help='endpoint for `payload` (run it on the other machine)')
    p.add_argument('--host', default='127.0.0.1', help='listen address; keep it local and use an SSH tunnel')
    p.add_argument('--port', type=int, default=8090)
    p.set_defaults(func=sink)
    p = sub.add_parser('payload', help='upload/download time vs size to a `sink`')
    p.add_argument('--url', required=True, help='the sink, e.g. http://127.0.0.1:8091 (a tunnel to the node)')
    p.add_argument('--label', required=True, help='name of the path, e.g. router-m8g (results folder)')
    p.add_argument('--sizes', type=int, nargs='+', default=PAYLOAD_SIZES, help='bytes')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--run-id')
    p.set_defaults(func=payload_sweep)
    p = sub.add_parser('netdev', help='network byte counters per host')
    p.add_argument('--hosts', nargs='+', help="SSH hosts (default: this machine; 'local' for this machine)")
    p.add_argument('--tag', default='', help='e.g. before-ect-load90')
    p.add_argument('--out', help='CSV to append to, e.g. results/movement/netdev.csv')
    p.set_defaults(func=netdev)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
