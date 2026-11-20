"""Consume prebuilt JSON bodies and RemoteSamples; buffer new measurement side CSVs.

Payload sizes match requests' json= serialization. CPU and process RSS observations
are sampled at most once per poll, so short peaks can be missed.
"""
import json
import threading
import time
from benchmark import append_rows


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

    def poll(self):
        """Poll once a second without repeating stale readings."""
        while not self.stop.wait(1): self.observe()

    def record(self, request_id, size, decision_us, slots):
        """Buffer one assignment duration, body size and effective chosen slot count."""
        self.requests.append(dict(run_id=self.run_id,id=request_id,req_bytes=size,
                                  decision_us=decision_us,node_slots=slots))

    def close(self):
        """Join polling before monitors close, then flush on completion or interruption."""
        self.stop.set(); self.thread.join()
        append_rows(self.directory/'requests_extra.csv',
                    ['run_id','id','req_bytes','decision_us','node_slots'],self.requests)
        append_rows(self.directory/'node_samples.csv',
                    ['run_id','t_s','node','cpu_pct','ram_gb'],self.samples)
