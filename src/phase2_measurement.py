"""Consume prebuilt JSON bodies and RemoteSamples; buffer new measurement side CSVs.

Payload sizes match requests' json= serialization. CPU and process RSS observations
are sampled at most once per poll, so short peaks can be missed. The router machine itself
is sampled too (node 'router': whole-machine CPU % and used RAM, Linux only), since it
builds every request body.
"""
import json
from pathlib import Path
import threading
import time
from benchmark import append_rows

EXTRA_FIELDS = ['run_id', 'id', 'req_bytes', 'decision_us', 'node_slots', 'response_bytes', 'encoding_ms']


def host_usage():
    """(busy, total) CPU ticks of this machine and its used RAM in GB, or None off Linux."""
    try:
        ticks = [int(x) for x in Path('/proc/stat').read_text().split('\n', 1)[0].split()[1:]]
        mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    except (OSError, ValueError):
        return None
    idle = ticks[3] + (ticks[4] if len(ticks) > 4 else 0)   # idle + iowait
    used_kib = int(mem['MemTotal'].split()[0]) - int(mem['MemAvailable'].split()[0])
    return sum(ticks) - idle, sum(ticks), used_kib / 1024 ** 2


def request_bytes(body):
    """Return UTF-8 JSON bytes using requests' default serialization options."""
    return len(json.dumps(body, allow_nan=False).encode('utf-8'))


class Measurements:
    """Buffer assignment timings and fresh monitor observations for one AWS replay."""
    def __init__(self, directory, run_id, monitors, start):
        """Skip pre-run samples and initialize the replay-relative observation clock."""
        self.directory, self.run_id, self.monitors, self.start = directory, run_id, monitors, start
        self.counts = {n:len(m.samples) for n,m in monitors.items()}
        self.requests, self.samples = [], []
        self.by_id = {}
        self.host = host_usage()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def observe(self):
        """Record the latest sample only when the monitor's sample count increases."""
        for node,monitor in self.monitors.items():
            count = len(monitor.samples)
            if count > self.counts[node]:
                sample = monitor.samples[count-1]
                self.samples.append(dict(run_id=self.run_id,t_s=time.perf_counter()-self.start,
                                         node=node,cpu_pct=sample.get('cpu_pct'),ram_gb=sample.get('ram_gb')))
                self.counts[node] = count
        now = host_usage()
        if now and self.host:
            busy, total = now[0] - self.host[0], now[1] - self.host[1]
            self.samples.append(dict(run_id=self.run_id, t_s=time.perf_counter()-self.start, node='router',
                                     cpu_pct=100 * busy / total if total else None, ram_gb=now[2]))
        self.host = now

    def poll(self):
        """Poll once a second without repeating stale readings."""
        while not self.stop.wait(1): self.observe()

    def record(self, request_id, size, decision_us, slots):
        """Buffer one assignment duration, body size and effective chosen slot count."""
        row = dict(run_id=self.run_id,id=request_id,req_bytes=size,
                   decision_us=decision_us,node_slots=slots)
        self.requests.append(row)
        self.by_id[request_id] = row

    def complete(self, request_id, record):
        """Add a finished request's response size and encoder time (patched llama-server) to its row."""
        row = self.by_id.get(request_id)
        if row is not None:
            row.update(response_bytes=record.get('response_bytes'), encoding_ms=record.get('encoding_ms'))

    def close(self):
        """Join polling before monitors close, then flush on completion or interruption."""
        self.stop.set(); self.thread.join()
        append_rows(self.directory/'requests_extra.csv', EXTRA_FIELDS, self.requests)
        append_rows(self.directory/'node_samples.csv',
                    ['run_id','t_s','node','cpu_pct','ram_gb'],self.samples)
