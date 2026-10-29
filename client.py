"""Benchmark driver: sends requests to the llama.cpp servers on the AWS instances, runs the
microbenchmark / macrobenchmark from monitor.py, and records the outputs.

On each instance:
    llama-server -m <model.gguf> --mmproj <mmproj.gguf> --host 0.0.0.0 --port 8080
    python monitor.py --pid $(pgrep -n llama-server) --port 9100

nodes.json lists the instances:
    [{"name": "c7g-xlarge", "url": "http://<ip>:8080", "agent_url": "http://<ip>:9100"}]
With "ssh" (a host alias from ~/.ssh/config), the client opens an SSH tunnel for the run instead:
the local ports in url / agent_url are forwarded to ports 8080 / 9100 on that host.
    [{"name": "c7a-2xlarge", "ssh": "mm-node1",
      "url": "http://localhost:8080", "agent_url": "http://localhost:9100"}]

Microbenchmark, one request at a time on every node (one run-table row per node):
    python client.py micro --run-id 04 --dataset data/image_1.jsonl --modality image --input-size "1 image"
Macrobenchmark, a load of requests spread over the nodes:
    python client.py macro --run-id m01 --dataset data/mixed.jsonl --num-requests 200 \
        --rate 0.5 --burstiness 0.5 --ttft-slo 2 --e2e-slo 15

Outputs:
    results/micro/<run-id>/<node>/{summary.json, requests.csv, samples_<node>.csv}
    results/macro/<run-id>/{summary.json, requests.csv, samples_<node>.csv}
    results/{micro,macro}/results.csv: one row per run (and per node, for micro)
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import random
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterator
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

import monitor

DEFAULT_DATASET_PATH = os.environ.get("DATASET_PATH", "data/requests.jsonl")

MODALITIES = ("text", "image", "audio")


@dataclass
class Request:
    request_id: str
    modality: str                                       # "text" | "image" | "audio"
    prompt: str                                         # instruction, or the article text for CNN/DailyMail
    media: list[str] = field(default_factory=list)      # image / audio file paths
    meta: dict[str, Any] = field(default_factory=dict)  # e.g. {"input_tokens": 512}, {"audio_sec": 10}


def load_requests(dataset_path: str = DEFAULT_DATASET_PATH,
                  modality: str | None = None,
                  limit: int | None = None) -> list[Request]:
    """Read a JSONL request manifest, one request per line, e.g.

        {"id": "coco-0001", "modality": "image", "prompt": "Describe the image.",
         "media": ["images/000000000139.jpg"], "meta": {"num_images": 1}}

    Relative media paths are resolved against the manifest's directory.
    """
    path = Path(dataset_path)
    loaded: list[Request] = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            mod = rec.get("modality", "text")
            if mod not in MODALITIES:
                raise ValueError(f"{path}:{i + 1}: unknown modality {mod!r}")
            if modality and mod != modality:
                continue
            media = [p if Path(p).is_absolute() else str(path.parent / p) for p in rec.get("media", [])]
            loaded.append(Request(str(rec.get("id", i)), mod, rec.get("prompt", ""), media, rec.get("meta", {})))
            if limit and len(loaded) >= limit:
                break
    if not loaded:
        raise ValueError(f"no requests found in {path}" + (f" for modality {modality!r}" if modality else ""))
    return loaded


# llama.cpp
class LlamaCppClient:
    """Streams chat completions from one llama-server through its OpenAI-compatible API."""

    def __init__(self, name: str, url: str, agent_url: str | None = None, ignore_eos: bool = False,
                 timeout: float = 600.0):
        self.name = name
        self.url = url.rstrip("/")
        self.agent_url = agent_url
        self.ignore_eos = ignore_eos
        self.timeout = timeout
        self.session = requests.Session()
        # llama-server sometimes drops a kept-alive connection before the next request arrives
        # ("Remote end closed connection without response"): resend once on a fresh connection
        self.session.mount("http://", HTTPAdapter(max_retries=Retry(
            total=1, connect=1, read=1, status=0, other=0, allowed_methods=None, raise_on_status=False)))

    def ping(self) -> None:
        self.session.get(f"{self.url}/health", timeout=10).raise_for_status()

    def model_id(self) -> str:
        return self.session.get(f"{self.url}/v1/models", timeout=10).json()["data"][0]["id"]

    def preprocess(self, req: Request) -> list[dict]:
        """Message content: each image / audio file base64-encoded, then the text prompt."""
        parts = []
        for path in req.media:
            data = base64.b64encode(Path(path).read_bytes()).decode()
            if req.modality == "audio":
                parts.append({"type": "input_audio", "input_audio": {"data": data, "format": Path(path).suffix[1:]}})
            else:
                mime = mimetypes.guess_type(path)[0] or "image/jpeg"
                parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
        parts.append({"type": "text", "text": req.prompt})
        return parts

    def stream(self, content: list[dict], max_new_tokens: int) -> Iterator[str | dict]:
        """Yield each streamed token, then llama.cpp's timings dict when the server sends one."""
        body = {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_new_tokens,
            "temperature": 0,
            "stream": True,
            "cache_prompt": False,  # no KV-cache reuse between repeated benchmark prompts
            "ignore_eos": self.ignore_eos,
        }
        with self.session.post(f"{self.url}/v1/chat/completions", json=body, stream=True,
                               timeout=self.timeout) as resp:
            if not resp.ok:  # keep llama.cpp's reason, e.g. "Failed to load image or audio file"
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            # chunk_size=None hands over each chunk as it arrives instead of waiting to fill a buffer
            for line in resp.iter_lines(chunk_size=None):
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                for choice in chunk.get("choices", []):
                    if choice.get("delta", {}).get("content"):
                        yield choice["delta"]["content"]
                if "timings" in chunk:
                    yield chunk["timings"]


# ssh tunnels
def port_open(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("localhost", port)) == 0


@contextmanager
def ssh_tunnels(specs: list[dict], ready_timeout: float = 60.0) -> Generator[None]:
    """Open an SSH tunnel to every node in nodes.json that has an "ssh" host, wait until each
    llama-server answers /health, and close the tunnels on exit.

    A local port that is already open (e.g. a tunnel started by hand) is used as is.
    """
    procs = []
    try:
        for n in specs:
            if not n.get("ssh"):
                continue
            forwards = [f"{urlsplit(url).port}:localhost:{remote}"
                        for url, remote in ((n["url"], n.get("server_port", 8080)),
                                            (n.get("agent_url"), n.get("agent_port", 9100)))
                        if url and not port_open(urlsplit(url).port)]
            if not forwards:
                print(f"{n['name']}: local ports already open, using the existing tunnel")
                continue
            print(f"{n['name']}: tunnel to {n['ssh']} ({', '.join(forwards)})")
            procs.append(subprocess.Popen(
                ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
                 *(arg for f in forwards for arg in ("-L", f)), n["ssh"]]))
        deadline = time.monotonic() + ready_timeout
        for n in specs:
            while True:
                if any(p.poll() is not None for p in procs):
                    raise RuntimeError("ssh exited; see its error above")
                try:
                    if requests.get(f"{n['url'].rstrip('/')}/health", timeout=5).ok:
                        break
                except requests.ConnectionError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError(f"{n['name']}: no llama-server at {n['url']} after {ready_timeout:.0f}s")
                time.sleep(1)  # 503 while the model is still loading
        yield
    finally:
        for p in procs:
            p.terminate()
            p.wait()


# output
def save_run(run_dir: Path, summary: dict, results: list[monitor.RequestResult],
             samples: dict[str, list[dict]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_csv(run_dir / "requests.csv", [
        {**{k: v for k, v in asdict(r).items() if k != "output_text"},
         "ttft_s": monitor.time_to_first_token(r), "tpot_s": monitor.time_per_output_token(r),
         "e2e_s": monitor.overall_latency(r), "network_overhead_s": monitor.network_overhead(r),
         "output_text": r.output_text}
        for r in results
    ])
    for node, rows in samples.items():
        write_csv(run_dir / f"samples_{node}.csv", rows)


def print_outputs(results: list[monitor.RequestResult]) -> None:
    """Show each distinct error, and the start of one generated reply per node and request."""
    errors: dict[str, int] = {}
    shown = set()
    for r in results:
        if not r.ok:
            errors[f"{r.node}: {r.error}"] = errors.get(f"{r.node}: {r.error}", 0) + 1
        elif (r.node, r.request_id) not in shown:
            shown.add((r.node, r.request_id))
            text = " ".join(r.output_text.split())
            print(f"  {r.node} {r.request_id} -> {text[:150]}{'...' if len(text) > 150 else ''}")
    for err, count in errors.items():
        print(f"  FAILED x{count} {err}")


def model_name(client: LlamaCppClient) -> str:
    return Path(client.model_id()).stem  # "/home/ubuntu/models/Qwen2.5-Omni-3B-Q8_0.gguf" -> "Qwen2.5-Omni-3B-Q8_0"


def summary_row(summary: dict, labels: dict) -> dict:
    """One line of results.csv."""
    lat = summary["latency_s"]
    return {
        "run_id": summary["run_id"],
        "timestamp_utc": summary["timestamp_utc"],
        **labels,
        "modality": summary["config"]["modality"] or "mixed",
        "input_size": summary["config"]["input_size"],
        "dataset": summary["config"]["dataset"],
        "max_new_tokens": summary["config"]["max_new_tokens"],
        "completed": summary["requests"]["completed"],
        "failed": summary["requests"]["failed"],
        "median_e2e_s": lat["e2e"]["p50"],
        "p95_e2e_s": lat["e2e"]["p95"],
        "mean_e2e_s": lat["e2e"]["mean"],
        "median_ttft_s": lat["ttft"]["p50"],
        "p95_ttft_s": lat["ttft"]["p95"],
        "median_tpot_s": lat["tpot"]["p50"],
        "median_decode_tok_s": summary["decode_tok_s"]["p50"],
        "median_network_overhead_s": lat["network_overhead"]["p50"],
        "req_s": summary["throughput"]["req_s"],
        "output_tok_s": summary["throughput"]["output_tok_s"],
        "slo_attainment": summary["slo_attainment"],
    }


def record(path: Path, row: dict) -> None:
    append_csv(path, row)
    print(" | ".join(f"{k} {fmt(v)}" for k, v in row.items()))


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: fmt(v) for k, v in r.items()} for r in rows)


def append_csv(path: Path, row: dict) -> None:
    """Add a row, rewriting the file so earlier rows stay readable when columns were added since."""
    old = []
    if path.exists():
        with open(path, newline="") as f:
            old = list(csv.DictReader(f))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys([*row, *(k for r in old for k in r)])), restval="")
        w.writeheader()
        w.writerows(old)
        w.writerow({k: fmt(v) for k, v in row.items()})


# main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("micro", "macro"))
    p.add_argument("--run-id", required=True, help="e.g. 04 for row 04 of the run table")
    p.add_argument("--nodes", default="nodes.json", help="JSON list of {name, url, agent_url}")
    p.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="JSONL request manifest")
    p.add_argument("--modality", choices=MODALITIES, help="only use requests of this modality")
    p.add_argument("--input-size", default="", help='label recorded with the run, e.g. "2048 tokens"')
    p.add_argument("--num-requests", type=int, help="measured requests (cycles the dataset; default: all)")
    p.add_argument("--warmup", type=int, default=2, help="unmeasured requests per modality and node")
    p.add_argument("--max-new-tokens", type=int, default=128, help="output cap")
    p.add_argument("--ignore-eos", action="store_true", help="always generate --max-new-tokens tokens")
    p.add_argument("--ttft-slo", type=float, help="TTFT target, seconds")
    p.add_argument("--e2e-slo", type=float, help="E2E latency target, seconds")
    p.add_argument("--concurrency", type=int, default=1, help="macro, closed loop: requests in flight")
    p.add_argument("--rate", type=float, help="macro, open loop: arrival rate, req/s")
    p.add_argument("--burstiness", type=float, default=1.0, help="macro, open loop: 1 = Poisson, < 1 = burstier")
    p.add_argument("--seed", type=int, default=0, help="macro: request order and arrival times")
    p.add_argument("--output-dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    specs = json.loads(Path(args.nodes).read_text())
    nodes = [LlamaCppClient(n["name"], n["url"], n.get("agent_url"), args.ignore_eos) for n in specs]
    dataset = load_requests(args.dataset, args.modality)
    reqs = [dataset[i % len(dataset)] for i in range(args.num_requests or len(dataset))]
    out_dir = Path(args.output_dir) / args.mode
    meta = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": vars(args)}

    with ssh_tunnels(specs):
        if args.mode == "micro":
            for node in nodes:
                summary, results, samples = monitor.microbenchmark(node, reqs, args.max_new_tokens, args.warmup,
                                                                   args.ttft_slo, args.e2e_slo)
                summary = {**meta, "model": node.model_id(), **summary}
                save_run(out_dir / args.run_id / node.name, summary, results, samples)
                print_outputs(results)
                res = summary["resources"].get(node.name)
                record(out_dir / "results.csv", summary_row(summary, {
                    "node": node.name,
                    "model": model_name(node),
                    "instance": res["host"]["instance_type"] if res else None,
                    "peak_ram_gb": res["memory_gb"]["peak"] if res else None,
                    "mean_cpu_pct": res["cpu_pct"]["proc_normalized"]["mean"] if res else None,
                }))
        else:
            random.Random(args.seed).shuffle(reqs)
            summary, results, samples = monitor.macrobenchmark(
                nodes, reqs, concurrency=args.concurrency, rate=args.rate, burstiness=args.burstiness,
                max_new_tokens=args.max_new_tokens, warmup=args.warmup, ttft_slo=args.ttft_slo,
                e2e_slo=args.e2e_slo, seed=args.seed)
            summary = {**meta, "models": {n.name: n.model_id() for n in nodes}, **summary}
            save_run(out_dir / args.run_id, summary, results, samples)
            print_outputs(results)
            record(out_dir / "results.csv", summary_row(summary, {
                "nodes": "+".join(n.name for n in nodes),
                "models": "+".join(dict.fromkeys(model_name(n) for n in nodes)),
                "load": " ".join(f"{k}={v}" for k, v in summary["load"].items()),
            }))

if __name__ == "__main__":
    main()
