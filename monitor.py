"""Measurement functions for the microbenchmark and macrobenchmark.

Microbenchmark: requests sent one at a time to a single node, for one modality and
input size (one row of the run table). Gives the per-request latency breakdown and
the node's CPU / RAM.

Macrobenchmark: a stream of requests spread over several nodes, either closed loop
(fixed concurrency) or open loop (arrival rate, steady or bursty). Gives throughput,
latency percentiles and SLO attainment overall, per node and per modality, plus each
node's CPU / RAM.

Measurements:
1. Metrics (per request, then mean / p50 / p95 over the run)
(a) E2E latency: from submission to last token (includes preprocessing)
(b) TTFT: from submission to first token (includes preprocessing)
(c) TPOT: (last token - first token) / (output tokens - 1)
(d) decode tok/s: 1 / TPOT
(e) network overhead: send -> first token, minus the server's prefill time

2. Run-level:
(a) throughput: completed requests / wall time, and output tokens / wall time
(b) peak RAM: max RSS of llama-server, sampled, plus the kernel high-water mark
(c) CPU %: mean and max CPU of llama-server, and system-wide utilization
(d) SLO attainment: fraction of requests meeting the TTFT / E2E targets

CPU and RAM are measured on each instance by the resource agent in this file:
    python monitor.py --pid $(pgrep -n llama-server) --port 9100

The client objects passed in need: .name, .agent_url (or None), .ping(),
.preprocess(request) -> payload, and .stream(payload, max_new_tokens), which yields
one str per streamed token and then, optionally, the server's timings dict.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import random
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Callable

import psutil

if TYPE_CHECKING:
    from client import LlamaCppClient, Request

GB = 1024 ** 3


# resources (run on the instance)
class ResourceSampler(threading.Thread):
    """Samples RSS and CPU% of a process tree in the background."""

    def __init__(self, pid: int, interval: float):
        super().__init__(daemon=True)
        self.root = psutil.Process(pid)
        self.interval = interval
        self.samples: list[dict] = []
        self._halt = threading.Event()
        self._procs: dict[int, psutil.Process] = {}
        self._t0 = 0.0

    def _tree(self) -> list[psutil.Process]:
        try:
            current = [self.root] + self.root.children(recursive=True)
        except psutil.NoSuchProcess:
            return []
        # Reuse Process objects so cpu_percent() measures since the previous sample.
        return [self._procs.setdefault(p.pid, p) for p in current]

    def run(self) -> None:
        self._t0 = time.perf_counter()
        for p in self._tree():
            p.cpu_percent(None)
        psutil.cpu_percent(None)
        while not self._halt.wait(self.interval):
            rss, cpu = 0, 0.0
            for p in self._tree():
                try:
                    rss += p.memory_info().rss
                    cpu += p.cpu_percent(None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.samples.append({
                "t_s": round(time.perf_counter() - self._t0, 4),
                "rss_gb": rss / GB,
                "proc_cpu_pct": cpu,
                "sys_cpu_pct": psutil.cpu_percent(None),
                "sys_mem_used_gb": psutil.virtual_memory().used / GB,
            })

    def stop(self) -> None:
        self._halt.set()
        self.join()


def kernel_peak_rss_gb(pid: int) -> float | None:
    """Kernel-tracked peak RSS (VmHWM, Linux only); catches spikes between samples."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024 / GB
    except OSError:
        pass
    return None


def reset_kernel_peak_rss(pid: int) -> bool:
    """Reset VmHWM to the current RSS (Linux 4.0+), so the kernel peak covers only the measured window."""
    try:
        with open(f"/proc/{pid}/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def ec2_instance_type(timeout: float = 0.5) -> str | None:
    """Ask the EC2 instance metadata service (IMDSv2); None when not on EC2."""
    base = "http://169.254.169.254/latest"
    try:
        tok_req = urllib.request.Request(f"{base}/api/token", method="PUT",
                                         headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        token = urllib.request.urlopen(tok_req, timeout=timeout).read().decode()
        req = urllib.request.Request(f"{base}/meta-data/instance-type",
                                     headers={"X-aws-ec2-metadata-token": token})
        return urllib.request.urlopen(req, timeout=timeout).read().decode()
    except Exception:
        return None


def host_info() -> dict:
    return {
        "instance_type": ec2_instance_type() or "unknown",
        "hostname": platform.node(),
        "arch": platform.machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "vcpus": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "total_ram_gb": psutil.virtual_memory().total / GB,
    }


def serve_agent(pid: int, port: int = 9100, interval: float = 0.1) -> None:
    """Serve ResourceSampler over HTTP: POST /start begins sampling `pid`, POST /stop returns the samples."""
    state: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path == "/start":
                if "sampler" in state:
                    state.pop("sampler").stop()
                state["hwm_reset"] = reset_kernel_peak_rss(pid)
                state["sampler"] = ResourceSampler(pid, interval)
                state["sampler"].start()
                body = {"ok": True}
            elif self.path == "/stop" and "sampler" in state:
                sampler = state.pop("sampler")
                sampler.stop()
                body = {"samples": sampler.samples, "kernel_peak_rss_gb": kernel_peak_rss_gb(pid),
                        "hwm_reset": state["hwm_reset"], "host": host_info()}
            else:
                self.send_error(404)
                return
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args) -> None:
            pass

    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


# resources (called from the client)
def _post_json(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, method="POST"), timeout=60) as resp:
        return json.loads(resp.read())


def start_resource_sampling(agent_url: str) -> None:
    _post_json(f"{agent_url.rstrip('/')}/start")


def stop_resource_sampling(agent_url: str) -> dict:
    return _post_json(f"{agent_url.rstrip('/')}/stop")


# requests
@dataclass
class RequestResult:
    request_id: str
    modality: str
    node: str
    ok: bool = False
    error: str = ""
    # client timestamps in seconds, relative to the start of the measured window
    submit_s: float = 0.0
    preprocessed_s: float | None = None
    first_token_s: float | None = None
    end_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int = 0
    # durations reported by the server (llama.cpp timings)
    server_prefill_s: float | None = None
    server_decode_s: float | None = None
    output_text: str = ""  # the generated reply, to check the model's answers afterwards


def run_request(client: LlamaCppClient, req: Request, max_new_tokens: int, t_origin: float) -> RequestResult:
    res = RequestResult(req.request_id, req.modality, client.name, submit_s=time.perf_counter() - t_origin)
    timings: dict = {}
    chunks: list[str] = []
    try:
        payload = client.preprocess(req)
        res.preprocessed_s = time.perf_counter() - t_origin
        for item in client.stream(payload, max_new_tokens):
            if isinstance(item, dict):
                timings = item
                continue
            res.end_s = time.perf_counter() - t_origin
            if res.first_token_s is None:
                res.first_token_s = res.end_s
            chunks.append(item)
    except Exception as e:  # keep the run going; failures are counted in the summary
        res.error = repr(e)
        return res
    res.input_tokens = timings.get("prompt_n")
    res.output_tokens = timings.get("predicted_n", len(chunks))  # one chunk can carry more than one token
    res.server_prefill_s = timings["prompt_ms"] / 1000 if "prompt_ms" in timings else None
    res.server_decode_s = timings["predicted_ms"] / 1000 if "predicted_ms" in timings else None
    res.output_text = "".join(chunks)
    res.ok = len(chunks) > 0
    res.error = "" if res.ok else "no tokens generated"
    return res


# load generation
def warm_up(client: LlamaCppClient, requests: list[Request], n: int, max_new_tokens: int) -> None:
    """Send n unmeasured requests per modality so one-time costs stay out of the measurements."""
    for modality in {r.modality for r in requests}:
        same = [r for r in requests if r.modality == modality]
        for i in range(n):
            run_request(client, same[i % len(same)], max_new_tokens, time.perf_counter())


def round_robin(clients: list[LlamaCppClient]) -> Callable[[Request], LlamaCppClient]:
    nodes = itertools.cycle(clients)
    return lambda req: next(nodes)


def arrival_times(n: int, rate: float, burstiness: float = 1.0, seed: int = 0) -> list[float]:
    """Send times for n requests at `rate` req/s. Gamma inter-arrivals:
    burstiness 1 = Poisson, < 1 = burstier, > 1 = more evenly spaced."""
    rng = random.Random(seed)
    t, times = 0.0, []
    for _ in range(n):
        times.append(t)
        t += rng.gammavariate(burstiness, 1 / (rate * burstiness))
    return times


def run_closed_loop(pick: Callable[[Request], LlamaCppClient], requests: list[Request], concurrency: int,
                    max_new_tokens: int, t_origin: float) -> list[RequestResult]:
    """Keep `concurrency` requests in flight until all requests are done."""
    def send(r: Request) -> RequestResult:
        return run_request(pick(r), r, max_new_tokens, t_origin)
    if concurrency == 1:
        return [send(r) for r in requests]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(send, requests))


def run_open_loop(pick: Callable[[Request], LlamaCppClient], requests: list[Request], arrivals: list[float],
                  max_new_tokens: int, t_origin: float) -> list[RequestResult]:
    """Send request i at t_origin + arrivals[i], however many requests are still in flight."""
    def send(i: int) -> RequestResult:
        time.sleep(max(0.0, t_origin + arrivals[i] - time.perf_counter()))
        return run_request(pick(requests[i]), requests[i], max_new_tokens, t_origin)
    with ThreadPoolExecutor(max_workers=len(requests)) as pool:
        return list(pool.map(send, range(len(requests))))


# stats
def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]."""
    xs = sorted(values)
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def describe(values: list[float | None]) -> dict:
    xs = [v for v in values if v is not None]
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {"n": len(xs), "mean": sum(xs) / len(xs), "p50": percentile(xs, 50),
            "p95": percentile(xs, 95), "min": min(xs), "max": max(xs)}


# metrics
def measure_cpu_utilization(samples: list[dict], vcpus: int) -> dict:
    """Mean/max CPU of the serving process (100 = one core, and normalized to all vCPUs) and system-wide."""
    return {
        "proc": describe([s["proc_cpu_pct"] for s in samples]),
        "proc_normalized": describe([s["proc_cpu_pct"] / vcpus for s in samples]),
        "system": describe([s["sys_cpu_pct"] for s in samples]),
    }


def measure_peak_ram(samples: list[dict], kernel_peak_gb: float | None) -> dict:
    """Max sampled RSS of the serving process (+ children) and the kernel high-water mark."""
    sampled = max((s["rss_gb"] for s in samples), default=None)
    return {
        "peak_rss_sampled": sampled,
        "peak_rss_kernel": kernel_peak_gb,
        "peak_sys_used": max((s["sys_mem_used_gb"] for s in samples), default=None),
        "peak": max((v for v in (sampled, kernel_peak_gb) if v is not None), default=None),
    }


def measure_throughput(results: list[RequestResult], wall_s: float) -> dict:
    """Completed requests / wall time, and output tokens / wall time."""
    ok = [r for r in results if r.ok]
    return {"req_s": len(ok) / wall_s, "output_tok_s": sum(r.output_tokens for r in ok) / wall_s}


def measure_slo_attainment(results: list[RequestResult], ttft_slo: float | None = None,
                           e2e_slo: float | None = None) -> float | None:
    """Fraction of requests (failures count as misses) meeting every given target, in seconds."""
    if not results or (ttft_slo is None and e2e_slo is None):
        return None
    met = sum(r.ok and (ttft_slo is None or time_to_first_token(r) <= ttft_slo)
              and (e2e_slo is None or overall_latency(r) <= e2e_slo) for r in results)
    return met / len(results)


def measure_rtt(client: LlamaCppClient, n: int = 20) -> dict:
    """Round-trip time of the node's health check: the network floor under TTFT."""
    rtts = []
    for _ in range(n):
        t = time.perf_counter()
        client.ping()
        rtts.append(time.perf_counter() - t)
    return describe(rtts)


def time_to_first_token(r: RequestResult) -> float | None:
    """Submission -> first token (includes preprocessing)."""
    return r.first_token_s - r.submit_s if r.ok else None


def time_per_output_token(r: RequestResult) -> float | None:
    """(last token - first token) / (output tokens - 1)."""
    return (r.end_s - r.first_token_s) / (r.output_tokens - 1) if r.ok and r.output_tokens > 1 else None


def overall_latency(r: RequestResult) -> float | None:
    """Submission -> last token (includes preprocessing)."""
    return r.end_s - r.submit_s if r.ok else None


def network_overhead(r: RequestResult) -> float | None:
    """Send -> first token minus the server's prefill time: network, HTTP and server queueing."""
    if not r.ok or r.server_prefill_s is None:
        return None
    return r.first_token_s - r.preprocessed_s - r.server_prefill_s


# benchmarks
def summarize_requests(results: list[RequestResult], wall_s: float, ttft_slo: float | None = None,
                       e2e_slo: float | None = None) -> dict:
    ok = [r for r in results if r.ok]
    tpot = [time_per_output_token(r) for r in ok]
    return {
        "requests": {"completed": len(ok), "failed": len(results) - len(ok)},
        "wall_s": wall_s,
        "latency_s": {
            "e2e": describe([overall_latency(r) for r in ok]),
            "ttft": describe([time_to_first_token(r) for r in ok]),
            "tpot": describe(tpot),
            "preprocess": describe([r.preprocessed_s - r.submit_s for r in ok]),
            "server_prefill": describe([r.server_prefill_s for r in ok]),
            "server_decode": describe([r.server_decode_s for r in ok]),
            "network_overhead": describe([network_overhead(r) for r in ok]),
        },
        "decode_tok_s": describe([1 / t for t in tpot if t]),
        "throughput": measure_throughput(results, wall_s),
        "tokens": {"input": describe([r.input_tokens for r in ok]), "output": describe([r.output_tokens for r in ok])},
        "slo_attainment": measure_slo_attainment(results, ttft_slo, e2e_slo),
    }


def summarize_resources(agent_result: dict) -> dict:
    """CPU and peak RAM from one agent's /stop response."""
    samples = agent_result["samples"]
    kernel_peak = agent_result["kernel_peak_rss_gb"] if agent_result["hwm_reset"] else None  # else lifetime peak
    return {
        "host": agent_result["host"],
        "cpu_pct": measure_cpu_utilization(samples, agent_result["host"]["vcpus"]),
        "memory_gb": measure_peak_ram(samples, kernel_peak),
    }


def measured_run(clients: list[LlamaCppClient],
                 run: Callable[[float], list[RequestResult]]) -> tuple[list[RequestResult], float, dict]:
    """Sample resources on every node that has an agent while `run(t_origin)` sends the requests."""
    agents = {c.name: c.agent_url for c in clients if c.agent_url}
    for url in agents.values():
        start_resource_sampling(url)
    t_start = time.perf_counter()
    results = run(t_start)
    wall_s = time.perf_counter() - t_start
    return results, wall_s, {name: stop_resource_sampling(url) for name, url in agents.items()}


def microbenchmark(client: LlamaCppClient, requests: list[Request], max_new_tokens: int = 128, warmup: int = 2,
                   ttft_slo: float | None = None, e2e_slo: float | None = None,
                   rtt_pings: int = 20) -> tuple[dict, list[RequestResult], dict[str, list[dict]]]:
    """Send `requests` one at a time to a single node.

    Returns (summary, per-request results, resource samples per node).
    """
    rtt = measure_rtt(client, rtt_pings)
    warm_up(client, requests, warmup, max_new_tokens)
    results, wall_s, agents = measured_run(
        [client], lambda t0: run_closed_loop(lambda r: client, requests, 1, max_new_tokens, t0))
    summary = {
        "node": client.name,
        "network_rtt_s": rtt,
        **summarize_requests(results, wall_s, ttft_slo, e2e_slo),
        "resources": {name: summarize_resources(a) for name, a in agents.items()},
    }
    return summary, results, {name: a["samples"] for name, a in agents.items()}


def macrobenchmark(clients: list[LlamaCppClient], requests: list[Request],
                   pick: Callable[[Request], LlamaCppClient] | None = None, concurrency: int = 1,
                   rate: float | None = None, burstiness: float = 1.0, max_new_tokens: int = 128, warmup: int = 2,
                   ttft_slo: float | None = None, e2e_slo: float | None = None,
                   seed: int = 0) -> tuple[dict, list[RequestResult], dict[str, list[dict]]]:
    """Send `requests` across `clients` under load.

    Closed loop keeps `concurrency` requests in flight; giving `rate` (req/s) switches to
    open loop with gamma inter-arrivals (`burstiness` 1 = Poisson, < 1 = burstier).
    `pick(request) -> client` chooses the node: round-robin by default, the Phase 2 router plugs in here.

    Returns (summary, per-request results, resource samples per node).
    """
    pick = pick or round_robin(clients)
    for c in clients:
        warm_up(c, requests, warmup, max_new_tokens)
    if rate:
        arrivals = arrival_times(len(requests), rate, burstiness, seed)
        results, wall_s, agents = measured_run(
            clients, lambda t0: run_open_loop(pick, requests, arrivals, max_new_tokens, t0))
    else:
        results, wall_s, agents = measured_run(
            clients, lambda t0: run_closed_loop(pick, requests, concurrency, max_new_tokens, t0))
    summary = {
        "nodes": [c.name for c in clients],
        "load": {"rate": rate, "burstiness": burstiness} if rate else {"concurrency": concurrency},
        **summarize_requests(results, wall_s, ttft_slo, e2e_slo),
        "by_node": {c.name: summarize_requests([r for r in results if r.node == c.name], wall_s, ttft_slo, e2e_slo)
                    for c in clients},
        "by_modality": {m: summarize_requests([r for r in results if r.modality == m], wall_s, ttft_slo, e2e_slo)
                        for m in sorted({r.modality for r in results})},
        "resources": {name: summarize_resources(a) for name, a in agents.items()},
    }
    return summary, results, {name: a["samples"] for name, a in agents.items()}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Resource agent: run on each instance next to llama-server.")
    p.add_argument("--pid", type=int, required=True, help="llama-server PID")
    p.add_argument("--port", type=int, default=9100)
    p.add_argument("--interval", type=float, default=0.1, help="sampling period, seconds")
    a = p.parse_args()
    serve_agent(a.pid, a.port, a.interval)
