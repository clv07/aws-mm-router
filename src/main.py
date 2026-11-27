#!/usr/bin/env python3
"""Routing benchmark runner: replays a tracegen.py trace open loop over llama-server nodes (`run`) or a
simulated cluster (`sim`) with router.py's policies, so every policy sees exactly the same traffic.

  python src/main.py capacity --nodes nodes.json                   # predicted cost per node, capacity
  python src/main.py run --trace traces/<name>.csv --nodes nodes.json --policy ect --cluster C
  python src/main.py sim --trace traces/<name>.csv --nodes nodes.json   # every policy, simulated
"""
from __future__ import annotations

import logging
import csv
import argparse
from collections import deque
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import random
import threading
import time

import requests

# Phase 2 measurement: buffer side files without changing request result columns.
from phase2_measurement import Measurements, request_bytes

from benchmark import RUNS, RemoteSamples, append_rows, derived, median, payload, request_once, server_info
from router.router import (INF, POLICIES, QUEUE_POLICIES, Request, Router, fit_service_model, load_nodes,
                    node_model, read_csv)
from tracegen import DEFAULT_MIX, parse_mix

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from statistics import median as statistics_median
from router.calibrate import calibrate_node
from router.interference import InterferencePredictor

logger = logging.getLogger(__name__)


OUT_DIR = Path("results/routing")
REQUEST_FIELDS = ["run_id", "timestamp_utc", "cluster", "trace", "policy", "seed", "output_mode",
                  "id", "arrival_s", "sent_s", "node", "model", "node_inflight", "predicted_s",
                  "predicted_e2e_s", "run", "modality", "size", "input", "status", "error",
                  "finish_reason", "e2e_s", "ttft_s", "queue_s", "prompt_tokens", "output_tokens",
                  "prefill_tok_s", "decode_tok_s", "tpot_ms", "server_prompt_ms",
                  "server_predicted_ms"]
REQUEST_FIELDS += [
    "dispatch_s",
    "http_e2e_s",
    "router_queue_s",
    "server_queue_est_s",
    "deadline_s", "ttft_deadline_s", "tpot_slo_ms",
    "completion_slo_met", "ttft_slo_met", "tpot_slo_met", "overall_slo_met",
]
SUMMARY_FIELDS = ["run_id", "timestamp_utc", "cluster", "trace", "policy", "seed", "output_mode",
                  "offered_req_s", "group", "requests", "success", "failed", "median_e2e_s",
                  "p95_e2e_s", "p99_e2e_s", "median_ttft_s", "p95_ttft_s", "median_queue_s",
                  "req_s", "usd_per_1k_requests", "peak_ram_gb", "cpu_pct",
                  "completion_slo_attainment", "ttft_slo_attainment",
                  "tpot_slo_attainment", "overall_slo_attainment"]
SIM_FIELDS = ["run_id", "timestamp_utc", "trace", "policy", "seed", "slo_mult", "group", "requests",
              "median_e2e_s", "p95_e2e_s", "slo_attainment", "req_s", "usd_per_1k_requests",
              "decision_us"]
MODALITIES = ('chat', 'text', 'image', 'audio')


def utc_now():
    """Current UTC time as an ISO timestamp, to the second."""
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def new_run_id():
    """Run id: the current UTC time, e.g. 20261113T184453Z."""
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


# (a) prepare: trace, nodes, requests

def read_trace(path, n):
    """First n requests of a trace (all if n is None)."""
    trace = read_csv(path)[:n]
    if not trace:
        raise SystemExit(f'{path} has no requests')
    return trace


def validate_live_trace(trace, path):
    """Every live policy must use the same request-level SLO source and units.

    Only newly generated traces containing measured SLO thresholds are accepted.
    """
    required = ("id", "arrival_s", "run", "modality", "size", "input",
                "deadline_s", "ttft_deadline_s", "tpot_slo_ms")
    seen = set()
    for q in trace:
        missing = [key for key in required if q.get(key) in (None, "")]
        if missing:
            raise ValueError(
                f"{path}: request {q.get('id')} missing {missing}; "
                "generate a new trace with `python src/tracegen.py make --nodes nodes.json ...`"
            )
        if q["id"] in seen:
            raise ValueError(f"{path}: duplicate id {q['id']}")
        seen.add(q["id"])
        if q["run"] not in RUNS or RUNS[q["run"]][0] != q["modality"]:
            raise ValueError(f"{path}: invalid run/modality for id {q['id']}")
        for key in ("arrival_s", "deadline_s", "ttft_deadline_s", "tpot_slo_ms"):
            value = float(q[key])
            if not math.isfinite(value) or (value < 0 if key == "arrival_s" else value <= 0):
                raise ValueError(f"{path}: invalid {key} for id {q['id']}")
    return trace


def attach_live_slos(row, q):
    """Compute arrival-relative SLO verdicts; missing measurements fail the SLO."""
    row.update({k: float(q[k]) for k in (
        "deadline_s", "ttft_deadline_s", "tpot_slo_ms")})
    if row["status"] != "ok":
        outcomes = (False, False, False)
    else:
        def met(field, limit):
            try:
                value = float(row[field])
                return math.isfinite(value) and value <= limit
            except (ValueError, TypeError, KeyError):
                return False
        outcomes = (
            met("e2e_s", row["deadline_s"]),
            met("ttft_s", row["ttft_deadline_s"]),
            met("tpot_ms", row["tpot_slo_ms"]),
        )
    row.update(dict(zip(("completion_slo_met", "ttft_slo_met", "tpot_slo_met"), outcomes)))
    row["overall_slo_met"] = all(outcomes)
    return row


def print_model(nodes, costs, estimated, gains, mix):
    """Print each node's predicted cost per workload and capacity for this mix; return the cluster capacity."""
    print('predicted service time when alone, s (* = estimated from the other nodes):')
    short = {'chat': 'chat', 'text': 'txt', 'image': 'img', 'audio': 'aud'}
    head = [short[m] + ('' if m == 'chat' else str(size)) for m, size in RUNS.values()]
    print(f'  {"node":<16}' + ''.join(f'{h:>9}' for h in head) + f'{"gain":>7}{"cap req/s":>11}')
    total = 0.0
    for n in nodes:
        mean = sum(p * costs[n][r] for r, p in mix.items())
        cap = gains[n] / mean   # requests per second the node can sustain on this mix
        total += cap
        cells = ''.join(f'{costs[n][r]:>8.1f}' + ('*' if f'{n}:{r}' in estimated else ' ')
                        for r in RUNS)
        print(f'  {n:<16}{cells}{gains[n]:>7.2f}{cap:>11.3f}')
    print(f'  cluster capacity for this mix ≈ {total:.3f} req/s '
          f'(60% load = {0.6 * total:.3f}, 90% = {0.9 * total:.3f}); gain = slot_gain')
    return total


def capacity(args):
    """`capacity`: print predicted per-node costs and cluster capacity for a mix."""
    nodes = load_nodes(args.nodes)
    print_model(nodes, *node_model(nodes), parse_mix(args.mix))


def load_datasets(modalities):
    """One dataset loader per modality in the trace."""
    from loaders.coco import COCODataset
    from loaders.cnn_dailymail import CNNDailyMailDataset
    from loaders.dolly import DollyQADataset
    from loaders.librispeech import LibriSpeechDataset
    classes = {'chat': DollyQADataset, 'text': CNNDailyMailDataset,
               'image': COCODataset, 'audio': LibriSpeechDataset}
    return {m: classes[m]() for m in modalities}


def warm_up(nodes, datasets, output_mode):
    """One unrecorded request per node and modality (smallest size, input 0), nodes in parallel."""
    first = {}
    for run, (m, size) in RUNS.items():
        if m in datasets:
            first.setdefault(m, size)

    def warm(name):
        with requests.Session() as s:
            for m, size in first.items():
                record = request_once(s, nodes[name]['url'], payload(datasets[m], m, size, 0, output_mode))
                print(f'  warmup {name} {m}: {record["status"]} '
                      + (record['error'] or f'{record["e2e_s"]:.1f}s'), flush=True)
    threads = [threading.Thread(target=warm, args=(n,)) for n in nodes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def build_requests(args, trace, nodes):
    """HTTP body for every trace request, then warm up the nodes unless --no-warmup."""
    datasets = load_datasets({q['modality'] for q in trace})
    print('building requests', flush=True)
    bodies = {q['id']: payload(datasets[q['modality']], q['modality'], RUNS[q['run']][1],
                               int(q['input']), args.output) for q in trace}
    if not args.no_warmup:
        warm_up(nodes, datasets, args.output)
    return bodies


def collect_calibration_samples(nodes, datasets, output_mode):
    """
    Collect solo stage timing samples from every node.

    Returns:
        {
            node_name: [calibration_row, ...],
            ...
        }
    """
    samples = {}

    for node, cfg in nodes.items():
        logger.info("CALIBRATION_START node=%s", node)

        rows = []

        with requests.Session() as session:
            for run, (modality, size) in RUNS.items():
                body = payload(
                    datasets[modality],
                    modality,
                    size,
                    0,
                    output_mode,
                )

                result = request_once(
                    session,
                    cfg["url"],
                    body,
                )

                if result["status"] != "ok":
                    raise RuntimeError(
                        f"Calibration failed: node={node}, "
                        f"run={run}, error={result.get('error')}"
                    )

                prompt_tokens = result.get("prompt_tokens")
                output_tokens = result.get("output_tokens")
                prompt_ms = result.get("server_prompt_ms")
                decode_ms = result.get("server_predicted_ms")
                encoding_ms = result.get("encoding_ms")

                required = {
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "server_prompt_ms": prompt_ms,
                    "server_predicted_ms": decode_ms,
                }

                missing = [
                    key
                    for key, value in required.items()
                    if value is None
                ]

                if missing:
                    raise RuntimeError(
                        f"Missing calibration fields: "
                        f"node={node}, run={run}, "
                        f"fields={missing}"
                    )

                # Text and chat have no multimodal encoding.
                if modality in ("chat", "text"):
                    encoding_ms = 0.0
                elif encoding_ms is None:
                    raise RuntimeError(
                        f"Missing encoding_ms: "
                        f"node={node}, run={run}"
                    )

                prompt_tokens = float(prompt_tokens)
                output_tokens = float(output_tokens)
                prompt_ms = float(prompt_ms)
                decode_ms = float(decode_ms)
                encoding_ms = float(encoding_ms)

                values = (
                    prompt_tokens,
                    output_tokens,
                    prompt_ms,
                    decode_ms,
                    encoding_ms,
                )

                if not all(math.isfinite(v) for v in values):
                    raise RuntimeError(
                        f"Non-finite calibration value: "
                        f"node={node}, run={run}"
                    )

                if (
                    prompt_tokens <= 0
                    or output_tokens <= 0
                    or prompt_ms <= 0
                    or decode_ms <= 0
                    or encoding_ms < 0
                    or encoding_ms > prompt_ms
                ):
                    raise RuntimeError(
                        f"Invalid calibration timing: "
                        f"node={node}, run={run}"
                    )

                # The llama.cpp prompt timing includes
                # multimodal encoding in our patched build.
                prefill_ms = prompt_ms - encoding_ms

                row = {
                    "run": run,
                    "modality": modality,
                    "size": size,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "encoding_ms": encoding_ms,
                    "prefill_ms": prefill_ms,
                    "decode_ms": decode_ms,
                }

                rows.append(row)

                logger.info(
                    "CALIBRATION_SAMPLE node=%s run=%s "
                    "encoding=%.1fms prefill=%.1fms "
                    "decode=%.1fms",
                    node,
                    run,
                    encoding_ms,
                    prefill_ms,
                    decode_ms,
                )

        samples[node] = rows

        logger.info(
            "CALIBRATION_DONE node=%s samples=%d",
            node,
            len(rows),
        )

    return samples


def calibration_signature(nodes, output_mode):
    return {
        "version": 2,
        "output_mode": output_mode,
        "nodes": {
            name: {
                "url": cfg["url"],
                "model": cfg["model"],
                "slots": cfg["slots"],
                "server_build": cfg.get("server_build"),
                "context_per_slot": cfg.get("context_per_slot"),
                "calibration_id": cfg.get(
                    "calibration_id"
                ),
            }
            for name, cfg in nodes.items()
        },
    }


def valid_calibration_cache(cache, nodes, signature):
    if not isinstance(cache, dict):
        return False

    if cache.get("signature") != signature:
        return False

    params = cache.get("stage_params", {})
    prompt = cache.get("prompt_tokens_by_run", {})
    output = cache.get("output_tokens_by_run", {})

    required_params = (
        "image_encode_base",
        "image_encode_per_image",
        "audio_encode_base",
        "audio_encode_per_second",
        "prefill_s_per_token",
        "decode_s_per_token",
    )

    if not isinstance(params, dict):
        return False

    for node in nodes:
        p = params.get(node)

        if not isinstance(p, dict):
            return False

        for key in required_params:
            value = p.get(key)

            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                return False

        if (
            p["prefill_s_per_token"] <= 0
            or p["decode_s_per_token"] <= 0
        ):
            return False

        for size in (1, 2, 4):
            predicted = (
                p["image_encode_base"]
                + p["image_encode_per_image"] * size
            )
            if predicted < 0:
                return False

        for duration in (10, 40):
            predicted = (
                p["audio_encode_base"]
                + p["audio_encode_per_second"] * duration
            )
            if predicted < 0:
                return False

    for estimates in (prompt, output):
        if not isinstance(estimates, dict):
            return False

        for run in RUNS:
            value = estimates.get(run)

            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                return False

    return True


def load_or_calibrate(nodes, cache_path, output_mode):
    cache_path = Path(cache_path)
    signature = calibration_signature(
        nodes, output_mode
    )

    # 1. Try the cached parameters.
    if cache_path.exists():
        try:
            cache = json.loads(
                cache_path.read_text(
                    encoding="utf-8"
                )
            )

            if valid_calibration_cache(
                cache, nodes, signature
            ):
                logger.info(
                    "CALIBRATION_CACHE_HIT path=%s",
                    cache_path,
                )
                return cache

            logger.warning(
                "CALIBRATION_CACHE_INVALID path=%s",
                cache_path,
            )

        except (OSError, ValueError, TypeError) as exc:
            logger.warning(
                "CALIBRATION_CACHE_READ_ERROR error=%s",
                exc,
            )

    # 2. Cache miss: perform real calibration.
    logger.info("CALIBRATION_CACHE_MISS")

    datasets = load_datasets(
        {modality for modality, _ in RUNS.values()}
    )

    samples = collect_calibration_samples(
        nodes,
        datasets,
        output_mode,
    )

    # 3. Fit each node independently.
    stage_params = {}

    for node, rows in samples.items():
        stage_params[node] = calibrate_node(rows)

    # 4. Estimate tokens for each workload.
    prompt_tokens_by_run = {}
    output_tokens_by_run = {}

    for run in RUNS:
        matching = [
            row
            for rows in samples.values()
            for row in rows
            if row["run"] == run
        ]

        if not matching:
            raise RuntimeError(
                f"No calibration samples for run={run}"
            )

        prompt_tokens_by_run[run] = statistics_median(
            row["prompt_tokens"]
            for row in matching
        )

        output_tokens_by_run[run] = statistics_median(
            row["output_tokens"]
            for row in matching
        )

    cache = {
        "signature": signature,
        "stage_params": stage_params,
        "prompt_tokens_by_run": prompt_tokens_by_run,
        "output_tokens_by_run": output_tokens_by_run,
    }

    if not valid_calibration_cache(
        cache, nodes, signature
    ):
        raise RuntimeError(
            "Calibration produced invalid parameters"
        )

    # 5. Save atomically.
    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = cache_path.with_name(
        cache_path.name + ".tmp"
    )

    try:
        temp_path.write_text(
            json.dumps(cache, indent=2),
            encoding="utf-8",
        )

        os.replace(temp_path, cache_path)

    finally:
        if temp_path.exists():
            temp_path.unlink()

    logger.info(
        "CALIBRATION_CACHE_SAVED path=%s",
        cache_path,
    )

    return cache


def sim_requests(args, trace, nodes, costs, gains, model):
    """Simulator inputs: a Request (prompt tokens, deadline) per trace row and each one's actual alone time per node."""
    # Prompt tokens as logged when this trace ran on the cluster (tokenizing is deterministic, so
    # the router could count them on arrival), else the Phase 1 median.
    stem = Path(args.trace).stem
    logged = {}
    if (OUT_DIR / 'requests.csv').exists():
        for r in read_csv(OUT_DIR / 'requests.csv'):
            if r['trace'] == stem and r['status'] == 'ok' and r['prompt_tokens']:
                logged.setdefault(r['id'], float(r['prompt_tokens']))
    first = model[next(iter(nodes))]
    reqs = []
    for q in trace:
        arrival = float(q['arrival_s'])
        slo = args.slo * min(costs[n][q['run']] for n in nodes)   # x alone time on the fastest node
        reqs.append(Request(int(q['id']), q['run'], q['modality'],
                            logged.get(q['id'], first['prompt'].get(q['run'], 0.0)),
                            arrival, arrival + slo))
    # Actual alone time drawn once from Phase 1 requests (not the cost model) so every policy sees
    # the same service times; the Phase 1 median if a node has no sample.
    rng = random.Random(args.seed)
    actual = {(r.id, n): rng.choice(model[n]['samples'].get(r.run) or [costs[n][r.run]])
              for r in reqs for n in nodes}
    print(f'simulating {len(reqs)} requests of {args.trace} over '
          + ', '.join(f'{n} ({c["slots"]} slots, gain {gains[n]:.2f})' for n, c in nodes.items())
          + f'; SLO = {args.slo:g} x alone time on the fastest node; '
          f'{sum(q["id"] in logged for q in trace)} prompt token counts from {OUT_DIR}/requests.csv')
    return reqs, actual


# (b) route: live replay and simulator

def replay(args, nodes, trace, costs, gains, bodies, labels):
    """Send each request at its arrival time to the node router.Router picks; returns (rows, makespan, monitors)."""
    # Phase 2 measurement: count serialized payload bytes before the replay clock starts.
    body_bytes = {key: request_bytes(body) for key, body in bodies.items()}
    router = Router(args.policy, nodes, costs, gains, args.seed)
    write_lock = threading.Lock()
    rows = []

    worker_errors = []

    def send(q, node, arrival_s, dispatch_s, inflight, predicted):
        """Always record one terminal row, including unexpected HTTP exceptions."""
        sent_s = time.perf_counter() - start
        record = None
        try:
            with requests.Session() as session:
                record = request_once(session, nodes[node]['url'], bodies[q['id']])
        except Exception as exc:
            logger.exception("BASELINE_HTTP_ERROR policy=%s req=%s node=%s",
                             args.policy, q['id'], node)
            record = {
                'status': 'error', 'error': str(exc)[:180], 'finish_reason': None,
                'e2e_s': time.perf_counter() - start - sent_s,
                'ttft_s': None, 'decode_tok_s': None,
                'output_tokens': None, 'prompt_tokens': None,
                'server_prompt_ms': None, 'server_predicted_ms': None,
                'encoding_ms': None, 'response_bytes': None,
            }
        finally:
            finished_s = time.perf_counter() - start
            # Release the router assignment even if request_once raises.
            router.done(node, finished_s)

        # Auxiliary monitoring errors must not suppress the terminal request row.
        try:
            with write_lock:
                measurements.complete(q['id'], record)
        except Exception:
            logger.exception("BASELINE_MEASUREMENT_ERROR req=%s", q['id'])

        try:
            row = {**labels, 'timestamp_utc': utc_now(),
                   'id': q['id'], 'arrival_s': arrival_s, 'sent_s': sent_s, 'node': node,
                   'model': nodes[node]['model'], 'node_inflight': inflight,
                   'predicted_s': costs[node][q['run']], 'predicted_e2e_s': predicted,
                   'run': q['run'], 'modality': q['modality'], 'size': q['size'], 'input': q['input'],
                   **record, **derived(record)}
            server_queue_est_s = row['queue_s']
            row.update({
                'e2e_s': finished_s - arrival_s,
                'ttft_s': (sent_s - arrival_s + record['ttft_s'])
                          if record.get('ttft_s') is not None else None,
                'queue_s': max(0.0, dispatch_s - arrival_s),
                'dispatch_s': dispatch_s,
                'http_e2e_s': record.get('e2e_s'),
                'router_queue_s': max(0.0, dispatch_s - arrival_s),
                'server_queue_est_s': server_queue_est_s,
            })
            attach_live_slos(row, q)
            with write_lock:
                append_rows(OUT_DIR / 'requests.csv', REQUEST_FIELDS, [row])
                rows.append(row)
            logger.info("BASELINE_COMPLETE policy=%s req=%s node=%s status=%s",
                        args.policy, q['id'], node, record['status'])
            print(f"  {q['id']:>4} {q['run']} {q['modality']}={q['size']} -> {node} "
                  f"({inflight} in flight): {record['status']} "
                  + (record.get('error') or f"{record['e2e_s']:.1f}s, predicted {predicted:.1f}s"),
                  flush=True)
        except Exception as exc:
            # CSV/recording errors must reach the replay thread; a silent worker
            # exception would otherwise produce an incomplete, misleading summary.
            logger.exception("BASELINE_RESULT_ERROR policy=%s req=%s", args.policy, q['id'])
            with write_lock:
                worker_errors.append((q['id'], exc))

    with ExitStack() as stack:
        monitors = {n: stack.enter_context(RemoteSamples(c['ssh'])) for n, c in nodes.items() if c.get('ssh')}
        print(f'replaying {len(trace)} requests with policy {args.policy} over {list(nodes)}', flush=True)
        threads = []
        start = time.perf_counter()
        # Phase 2 measurement: skip pre-run samples and flush side files on interruption.
        measurements = Measurements(OUT_DIR, labels['run_id'], monitors, start)
        measurements.thread.start()
        try:
            for q in trace:
                wait = float(q['arrival_s']) - (time.perf_counter() - start)
                if wait > 0:
                    time.sleep(wait)
                now = time.perf_counter() - start
                # Phase 2 measurement: time only assignment to isolate router overhead.
                decision_start = time.perf_counter()
                node, inflight, predicted = router.assign(q['run'], now)
                dispatch_s = time.perf_counter() - start
                decision_us = (time.perf_counter() - decision_start) * 1e6
                # Phase 2 measurement: retain payload size and resolved slots for analysis.
                measurements.record(q['id'], body_bytes[q['id']], decision_us, nodes[node]['slots'])
                t = threading.Thread(target=send, args=(q, node, now, dispatch_s, inflight, predicted), daemon=True)
                t.start()
                threads.append(t)
            print('all requests sent; waiting for the last ones to finish', flush=True)
            for t in threads:
                t.join()
            with write_lock:
                failures = list(worker_errors)
                recorded = {r['id'] for r in rows}
            missing = [q['id'] for q in trace if q['id'] not in recorded]
            if failures or missing:
                raise RuntimeError(
                    f"Baseline replay incomplete: {len(recorded)}/{len(trace)} "
                    f"request results recorded; missing={missing}; "
                    f"worker_errors={[qid for qid, _ in failures]}"
                )
            makespan = time.perf_counter() - start
        finally:
            # Phase 2 measurement: stop polling after makespan and before monitor teardown.
            measurements.close()
    return rows, makespan, monitors


def build_stage_observation(request, result):
    """Extract valid stage timings from a successful instrumented HTTP call.

    Returns None when a required measurement is absent or inconsistent.
    This does not update either predictor; that belongs to Step 5.3.
    """
    if result.get("status") != "ok":
        return None
    if request.run not in RUNS:
        return None
    modality, size = RUNS[request.run]
    if modality != request.modality:
        return None

    encoding_ms = result.get("encoding_ms")
    if modality in ("chat", "text"):
        encoding_ms = 0.0
    values = (
        encoding_ms,
        result.get("server_prompt_ms"),
        result.get("server_predicted_ms"),
        result.get("prompt_tokens"),
        result.get("output_tokens"),
    )
    try:
        if any(value is None for value in values):
            return None
        encoding_ms, prompt_ms, decode_ms, prompt_tokens, output_tokens = (
            float(value) for value in values
        )
    except (ValueError, TypeError):
        return None
    if not all(math.isfinite(value) for value in (
        encoding_ms, prompt_ms, decode_ms, prompt_tokens, output_tokens
    )):
        return None
    if (encoding_ms < 0 or prompt_ms <= 0 or decode_ms <= 0 or
            encoding_ms >= prompt_ms or prompt_tokens <= 0 or output_tokens <= 0):
        return None

    prefill_ms = prompt_ms - encoding_ms
    return {
        "modality": modality,
        "image_count": int(size) if modality == "image" else 0,
        "audio_duration_s": float(size) if modality == "audio" else 0.0,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "encoding_ms": encoding_ms,
        "prompt_ms": prompt_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "encoding_s": encoding_ms / 1000.0,
        "prefill_s": prefill_ms / 1000.0,
        "decode_s": decode_ms / 1000.0,
    }


def replay_adaptive(args, nodes, trace, router, bodies, labels):
    """
    Live Adaptive Replay.

    - Replay real-time trace arrivals
    - Dispatch only when node slots are available
    - Execute real llama.cpp HTTP requests
    - Reschedule after each completion
    - Record results and Phase 2 measurements
    """
    trace = sorted(
        trace,
        key=lambda q: float(q["arrival_s"]),
    )

    by_id = {
        int(q["id"]): q
        for q in trace
    }

    if len(by_id) != len(trace):
        raise ValueError("Duplicate request IDs")

    if not trace:
        raise ValueError("Empty trace")

    body_bytes = {
        str(qid): request_bytes(body)
        for qid, body in bodies.items()
    }

    total = len(trace)
    completed = set()
    rows = []

    condition = threading.Condition()
    measurement_lock = threading.Lock()
    fatal_errors = []

    max_workers = sum(router.slots.values())

    if max_workers <= 0:
        raise ValueError("No HTTP worker slots")

    with ExitStack() as stack:
        monitors = {
            node: stack.enter_context(
                RemoteSamples(cfg["ssh"])
            )
            for node, cfg in nodes.items()
            if cfg.get("ssh")
        }

        start = time.perf_counter()

        measurements = Measurements(
            OUT_DIR,
            labels["run_id"],
            monitors,
            start,
        )
        measurements.thread.start()

        def clock():
            return time.perf_counter() - start

        def report_fatal(exc):
            logger.exception(
                "ADAPTIVE_WORKER_FATAL",
                exc_info=(
                    type(exc),
                    exc,
                    exc.__traceback__,
                ),
            )
            with condition:
                fatal_errors.append(exc)
                condition.notify_all()

        def get_snapshot(request, node):
            with router.lock:
                record = router.active[node][request.id]

                return {
                    "dispatch_s": record["dispatch_s"],
                    "predicted_finish":
                        record["predicted_finish"],
                    "inflight": len(router.active[node]),
                    "base_prediction": record.get("base_prediction"),
                    "expected_output_tokens": record.get(
                        "expected_output_tokens"
                    ),
                }

        def write_result(
            request,
            node,
            result,
            snapshot,
            sent_s,
            finished_s,
        ):
            q = by_id[request.id]

            dispatch_s = snapshot["dispatch_s"]
            predicted_s = (
                snapshot["predicted_finish"] - dispatch_s
            )

            router_queue_s = max(
                0.0,
                dispatch_s - request.arrival,
            )

            router_e2e_s = max(
                0.0,
                finished_s - request.arrival,
            )

            http_ttft = result.get("ttft_s")

            router_ttft_s = (
                sent_s - request.arrival + http_ttft
                if http_ttft is not None
                else None
            )

            # The original derived() estimates server
            # overhead from HTTP/server timings.
            metrics = derived(result)

            row = {
                **labels,
                "timestamp_utc": utc_now(),
                "id": q["id"],
                "arrival_s": request.arrival,
                "sent_s": sent_s,
                "node": node,
                "model": nodes[node]["model"],
                "node_inflight": snapshot["inflight"],
                "predicted_s": predicted_s,
                "predicted_e2e_s": (
                    predicted_s + router_queue_s
                ),
                "run": q["run"],
                "modality": q["modality"],
                "size": q["size"],
                "input": q["input"],
                **result,
                **metrics,

                # Router-visible latencies.
                "e2e_s": router_e2e_s,
                "ttft_s": router_ttft_s,
                "queue_s": router_queue_s,

                # Separate router and HTTP measurements.
                "dispatch_s": dispatch_s,
                "http_e2e_s": result.get("e2e_s"),
                "router_queue_s": router_queue_s,
                "server_queue_est_s": metrics.get("queue_s"),
            }

            attach_live_slos(row, q)

            with condition:
                if request.id in completed:
                    raise RuntimeError(
                        f"Duplicate completion: {request.id}"
                    )

                append_rows(
                    OUT_DIR / "requests.csv",
                    REQUEST_FIELDS,
                    [row],
                )

                rows.append(row)
                completed.add(request.id)
                condition.notify_all()

            logger.info(
                "RESULT req=%s node=%s status=%s "
                "queue=%.3fs e2e=%.3fs",
                request.id,
                node,
                result["status"],
                router_queue_s,
                router_e2e_s,
            )

        def submit_assignments(assignments, decision_us):
            for request, node in assignments:
                snapshot = get_snapshot(request, node)
                qid = str(request.id)

                with measurement_lock:
                    measurements.record(
                        qid,
                        body_bytes[qid],
                        decision_us,
                        router.slots[node],
                    )

                logger.info(
                    "DISPATCH req=%s node=%s t=%.3f",
                    request.id,
                    node,
                    snapshot["dispatch_s"],
                )

                executor.submit(
                    execute_request,
                    request,
                    node,
                    snapshot,
                )

        def execute_request(request, node, snapshot):
            try:
                sent_s = clock()
                # Real Router tracks HTTP windows. Legacy mock routers may
                # only implement dispatch_adaptive(), so keep them compatible.
                if hasattr(router, "mark_http_start"):
                    router.mark_http_start(request, node, sent_s)

                logger.info(
                    "HTTP_START req=%s node=%s t=%.3f",
                    request.id,
                    node,
                    sent_s,
                )

                try:
                    with requests.Session() as session:
                        result = request_once(
                            session,
                            nodes[node]["url"],
                            bodies[str(request.id)],
                        )

                except Exception as exc:
                    result = {
                        "status": "error",
                        "error": str(exc)[:180],
                        "finish_reason": None,
                        "e2e_s": None,
                        "ttft_s": None,
                        "decode_tok_s": None,
                        "output_tokens": None,
                        "prompt_tokens": None,
                        "server_prompt_ms": None,
                        "server_predicted_ms": None,
                    }

                finished_s = clock()
                success = result["status"] == "ok"
                observation = build_stage_observation(request, result)

                actual_s = (
                    finished_s - snapshot["dispatch_s"]
                )

                logger.info(
                    "HTTP_DONE req=%s node=%s "
                    "success=%s actual=%.3fs",
                    request.id,
                    node,
                    success,
                    actual_s,
                )

                # Completion may reserve slots for
                # waiting requests.
                decision_start = time.perf_counter()

                assignments = router.dispatch_adaptive(
                    now=finished_s,
                    finished=(
                        request,
                        node,
                        actual_s,
                        success,
                    ),
                )

                decision_us = (
                    time.perf_counter() - decision_start
                ) * 1e6

                overlaps = (
                    router.pop_completed_overlap(request.id)
                    if hasattr(router, "pop_completed_overlap") else []
                )
                # Update predictions without changing the old CSV schema.
                # A failed HTTP request cannot train on partial output.
                if success and hasattr(router, "apply_online_feedback"):
                    feedback = router.apply_online_feedback(
                        request=request,
                        node=node,
                        observation=observation,
                        overlaps=overlaps,
                        base_prediction=snapshot.get("base_prediction"),
                        expected_output_tokens=snapshot.get(
                            "expected_output_tokens"
                        ),
                        actual_output_tokens=result.get("output_tokens"),
                    )
                    logger.info(
                        "ONLINE_FEEDBACK req=%s node=%s base=%s "
                        "interference=%s output=%s peers=%d",
                        request.id, node,
                        feedback["base_updated"],
                        feedback["interference_updated"],
                        feedback["output_updated"], len(overlaps),
                    )
                else:
                    logger.debug(
                        "ONLINE_FEEDBACK_SKIPPED req=%s node=%s "
                        "success=%s stages=%s",
                        request.id, node, success,
                        observation is not None,
                    )

                with measurement_lock:
                    measurements.complete(
                        str(request.id),
                        result,
                    )

                write_result(
                    request,
                    node,
                    result,
                    snapshot,
                    sent_s,
                    finished_s,
                )

                submit_assignments(
                    assignments,
                    decision_us,
                )

            except Exception as exc:
                report_fatal(exc)

        try:
            with ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="adaptive-http",
            ) as executor:

                # -------- Arrival Loop --------

                for q in trace:
                    with condition:
                        if fatal_errors:
                            raise RuntimeError(
                                "Adaptive worker failed"
                            ) from fatal_errors[0]

                    target_s = float(q["arrival_s"])
                    delay_s = target_s - clock()

                    if delay_s > 0:
                        time.sleep(delay_s)

                    now = clock()

                    request = Request(
                        id=int(q["id"]),
                        run=q["run"],
                        modality=q["modality"],
                        prompt_tokens=(
                            router.prompt_tokens_by_run[
                                q["run"]
                            ]
                        ),
                        arrival=now,
                        deadline=(
                            now + float(q["deadline_s"])
                        ),
                    )

                    # Trace is the sole source of SLO.
                    router.request_slos[request.id] = {
                        "ttft_s": float(
                            q["ttft_deadline_s"]
                        ),
                        "tpot_s": (
                            float(q["tpot_slo_ms"]) / 1000
                        ),
                    }

                    decision_start = time.perf_counter()

                    assignments = router.dispatch_adaptive(
                        now=now,
                        arrived=request,
                    )

                    decision_us = (
                        time.perf_counter() - decision_start
                    ) * 1e6

                    submit_assignments(
                        assignments,
                        decision_us,
                    )

                logger.info(
                    "ALL_ARRIVED total=%d",
                    total,
                )

                # -------- Completion Wait --------

                deadline = clock() + args.max_drain_s

                while True:
                    with condition:
                        if fatal_errors:
                            raise RuntimeError(
                                "Adaptive worker failed"
                            ) from fatal_errors[0]

                        if len(completed) == total:
                            break

                    # Identify an undrainable queue.
                    with router.lock:
                        active_count = sum(
                            len(v)
                            for v in router.active.values()
                        )
                        queued_count = len(router.queue)

                    if active_count == 0 and queued_count > 0:
                        raise RuntimeError(
                            "Central queue has pending requests "
                            "but no runnable assignments"
                        )

                    remaining = deadline - clock()

                    if remaining <= 0:
                        raise TimeoutError(
                            "Adaptive replay drain timeout"
                        )

                    with condition:
                        condition.wait(
                            timeout=min(remaining, 0.5)
                        )

            makespan = clock()

        finally:
            measurements.close()

    return rows, makespan, monitors


def setup_logging(level=logging.INFO, policy="router", run_id=None):
    """One log per invocation: logs/<policy>_<run_id>.log."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or new_run_id()
    path = log_dir / f"{policy}_{run_id}.log"
    # Avoid mixing logs if two invocations happen within the same second.
    suffix = 1
    while path.exists():
        path = log_dir / f"{policy}_{run_id}_{suffix}.log"
        suffix += 1
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path, mode="w", encoding="utf-8")],
        force=True,
    )
    logger.info("RUN_LOG path=%s policy=%s run_id=%s", path, policy, run_id)
    return path


def run(args):
    """`run`: replay a trace over the live nodes with one policy, then evaluate."""
    if args.policy == "adaptive":
        return run_adaptive(args)
    nodes = load_nodes(args.nodes)
    trace = validate_live_trace(read_trace(args.trace, args.requests), args.trace)
    for cfg in nodes.values():
        cfg['model'], _, slots, _ = server_info(cfg['url'])
        cfg['slots'] = slots or 1
    costs, estimated, gains = node_model(nodes)
    mix = {r: sum(q['run'] == r for q in trace) / len(trace) for r in RUNS}
    offered = len(trace) / max(float(trace[-1]['arrival_s']), 1e-9)
    cap = print_model(nodes, costs, estimated, gains, mix)
    print(f'trace {args.trace}: {len(trace)} requests, offered {offered:.3f} req/s '
          f'= {offered / cap:.0%} of predicted capacity', flush=True)
    bodies = build_requests(args, trace, nodes)

    labels = {'run_id': getattr(args, 'run_id', None) or new_run_id(), 'cluster': args.cluster, 'trace': Path(args.trace).stem,
              'policy': args.policy, 'seed': args.seed, 'output_mode': args.output}
    rows, makespan, monitors = replay(args, nodes, trace, costs, gains, bodies, labels)
    evaluate_run(rows, nodes, labels, offered, makespan, monitors)

def run_adaptive(args):
    """
    Initialize and execute the Adaptive Router
    using cached or freshly collected stage calibration.
    """

    # 1. Load node configuration directly.
    # Do not require Phase 1 profile CSVs.
    nodes = json.loads(
        Path(args.nodes).read_text(
            encoding="utf-8"
        )
    )

    if not nodes:
        raise ValueError("No nodes configured")

    # 2. Check the actual llama.cpp servers.
    for node, cfg in nodes.items():
        if "url" not in cfg:
            raise ValueError(
                f"Missing URL for node {node}"
            )

        model, build, slots, context = server_info(
            cfg["url"]
        )

        cfg["model"] = model
        cfg["slots"] = int(slots or 1)
        cfg["server_build"] = build
        cfg["context_per_slot"] = context

    # 3. Read the trace.
    trace = read_trace(
        args.trace,
        args.requests,
    )

    validate_live_trace(trace, args.trace)

    # 4. Load calibration or collect new samples.
    cache = load_or_calibrate(
        nodes,
        args.adaptive_config,
        args.output,
    )

    stage_params = cache["stage_params"]

    prompt_estimates = cache[
        "prompt_tokens_by_run"
    ]

    output_estimates = cache[
        "output_tokens_by_run"
    ]

    # 5. Build real HTTP payloads.
    bodies = build_requests(
        args,
        trace,
        nodes,
    )

    # 6. Initialize Step 3 Router.
    # Legacy costs are not used by stage_cost().
    costs = {
        node: {}
        for node in nodes
    }

    gains = {
        node: 1.0
        for node in nodes
    }

    router = Router(
        policy="adaptive",
        nodes=nodes,
        costs=costs,
        gains=gains,
        seed=args.seed,
        stage_params=stage_params,
        interference_predictor=(
            InterferencePredictor()
        ),
    )

    # 7. Initialize token estimates.
    router.prompt_tokens_by_run = {
        run: float(value)
        for run, value in prompt_estimates.items()
    }

    router.output_estimates = {
        run: float(value)
        for run, value in output_estimates.items()
    }

    # Request-level SLO, populated at arrival.
    router.request_slos = {}

    # 8. Experiment labels.
    labels = {
        "run_id": getattr(args, 'run_id', None) or new_run_id(),
        "cluster": args.cluster,
        "trace": Path(args.trace).stem,
        "policy": "adaptive",
        "seed": args.seed,
        "output_mode": args.output,
    }

    logger.info(
        "ADAPTIVE_START nodes=%s requests=%d",
        list(nodes),
        len(trace),
    )

    # 9. Execute the real workload.
    rows, makespan, monitors = replay_adaptive(
        args,
        nodes,
        trace,
        router,
        bodies,
        labels,
    )

    # 10. Evaluate using the original function.
    last_arrival = max(
        float(q["arrival_s"])
        for q in trace
    )

    offered = (
        len(trace) / last_arrival
        if last_arrival > 0
        else 0.0
    )

    evaluate_run(
        rows,
        nodes,
        labels,
        offered,
        makespan,
        monitors,
    )

    logger.info(
        "ADAPTIVE_FINISH completed=%d "
        "makespan=%.3fs",
        len(rows),
        makespan,
    )


def sim_policy(policy, nodes, costs, gains, model, reqs, actual, seed):
    """Replay reqs on simulated nodes; returns ({id: {node, dispatch, finish}}, decision times in s).

    A node runs up to `slots` requests at once: alone at full speed, k at once at slot_gain / k
    each. Commit-on-arrival policies queue at the node; the slo policy queues at the router."""
    router = Router(policy, nodes, costs, gains, seed, model)
    running = {n: {} for n in nodes}   # id -> remaining seconds of work at full speed
    waiting = {n: deque() for n in nodes}
    by_id = {r.id: r for r in reqs}
    arrivals = deque(sorted(reqs, key=lambda r: r.arrival))
    log, decisions, now = {}, [], 0.0

    def rate(n):
        return 1.0 if len(running[n]) <= 1 else gains[n] / len(running[n])

    def start(r, n):
        running[n][r.id] = actual[r.id, n]
        log[r.id] = {'node': n, 'dispatch': now}

    def timed(fn, *a):
        t0 = time.perf_counter()
        out = fn(*a)
        decisions.append(time.perf_counter() - t0)
        return out

    while arrivals or any(running.values()):
        finish, who = INF, None
        for n in nodes:
            if running[n]:
                i = min(running[n], key=running[n].get)
                t = now + running[n][i] / rate(n)
                if t < finish:
                    finish, who = t, (n, i)
        arrival = arrivals[0].arrival if arrivals else INF
        t = min(finish, arrival)
        for n in nodes:
            k = rate(n) * (t - now)
            for i in running[n]:
                running[n][i] -= k
        now = t
        if finish <= arrival:
            n, i = who
            del running[n][i]
            log[i]['finish'] = now
            if policy in POLICIES:
                router.done(n, now)
                if waiting[n]:
                    start(waiting[n].popleft(), n)
            else:
                for r, m in timed(router.dispatch, now, None,
                                  (by_id[i], n, now - log[i]['dispatch'], True)):
                    start(r, m)
        else:
            r = arrivals.popleft()
            if policy in POLICIES:
                n = timed(router.assign, r.run, now)[0]
                if len(running[n]) < nodes[n]['slots']:
                    start(r, n)
                else:
                    waiting[n].append(r)
            else:
                for q, m in timed(router.dispatch, now, r):
                    start(q, m)
    return log, decisions


def simulate(args):
    """`sim`: simulate the trace with each policy over the node profiles, then evaluate."""
    nodes = load_nodes(args.nodes)
    slots = {k: int(v) for k, _, v in (p.partition('=') for p in args.slots.split(',') if p)}
    for name, cfg in nodes.items():
        cfg['slots'] = slots.get(name, 2)
    costs, _estimated, gains = node_model(nodes)
    model = fit_service_model(nodes)
    trace = read_trace(args.trace, args.requests)
    policies = args.policies.split(',')
    for p in policies:
        if p not in POLICIES + QUEUE_POLICIES:
            raise SystemExit(f'unknown policy {p!r}: use {", ".join(POLICIES + QUEUE_POLICIES)}')
    reqs, actual = sim_requests(args, trace, nodes, costs, gains, model)

    run_id = new_run_id()
    results = [(p, *sim_policy(p, nodes, costs, gains, model, reqs, actual, args.seed)) for p in policies]
    # Phase 2 measurement: export each completed replay before summary evaluation.
    for policy, log, decisions in results:
        # Phase 2 measurement: export service timestamps and sampled solo costs without changing scheduling.
        export_rows = []
        for r in reqs:
            e = log[r.id]
            export_rows.append(dict(source='sim', run_id=run_id, trace=Path(args.trace).stem, policy=policy,
                                    seed=args.seed, slo_mult=args.slo, id=r.id, run=r.run,
                                    modality=r.modality, arrival_s=r.arrival, node=e['node'],
                                    node_slots=nodes[e['node']]['slots'], start_s=e['dispatch'],
                                    finish_s=e['finish'], solo_s=actual[r.id, e['node']],
                                    sim_deadline_s=r.deadline, predicted_finish_s=None))
        # Phase 2 measurement: preserve the optional forecast column as empty; no sim_policy edit.
        append_rows(OUT_DIR / 'sim_requests.csv', list(export_rows[0]), export_rows)
        # Phase 2 measurement: retain every timed call, including dispatches that sent nothing.
        append_rows(OUT_DIR / 'sim_decisions.csv', ['run_id', 'policy', 'n', 'duration_us'],
                    [dict(run_id=run_id, policy=policy, n=i, duration_us=d * 1e6)
                     for i, d in enumerate(decisions)])
    evaluate_sim(results, reqs, nodes, args, run_id)


# (c) evaluate
# EVALUATION ENTRY POINTS: evaluate_run (live `run`) and evaluate_sim (`sim`) turn per-request
# results into the summary rows. Evaluation scripts plug in here.

def percentile(values, q):
    """Linear-interpolated q-quantile (0-1) of values; None if empty."""
    xs = sorted(values)
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def cluster_price(nodes):
    """Total $/hour of the nodes, and whether every node has a price."""
    return (sum(c.get('price_per_hour') or 0 for c in nodes.values()),
            all(c.get('price_per_hour') for c in nodes.values()))


def evaluate_run(rows, nodes, labels, offered, makespan, monitors):
    """Summary rows of a live run (all requests, each modality, each node), appended to summary.csv and printed."""
    price, has_price = cluster_price(nodes)
    groups = {'all': rows,
              **{m: [r for r in rows if r['modality'] == m] for m in MODALITIES},
              **{n: [r for r in rows if r['node'] == n] for n in nodes}}
    summaries = []
    for group, members in groups.items():
        if not members:
            continue
        ok = [r for r in members if r['status'] == 'ok']

        def valid(field):
            return [r[field] for r in ok if r[field] is not None]
        cpu, ram = monitors[group].metrics() if group in monitors else (None, None)
        def attainment(key):
            return sum(r.get(key) is True for r in members) / len(members)
        summaries.append({
            **labels, 'timestamp_utc': utc_now(),
            'offered_req_s': offered, 'group': group, 'requests': len(members),
            'success': len(ok), 'failed': len(members) - len(ok),
            'median_e2e_s': median(valid('e2e_s')), 'p95_e2e_s': percentile(valid('e2e_s'), .95),
            'p99_e2e_s': percentile(valid('e2e_s'), .99), 'median_ttft_s': median(valid('ttft_s')),
            'p95_ttft_s': percentile(valid('ttft_s'), .95), 'median_queue_s': median(valid('queue_s')),
            'req_s': len(ok) / makespan,
            'usd_per_1k_requests': price * makespan / 3600 / len(ok) * 1000
                                   if group == 'all' and has_price and ok else None,
            'peak_ram_gb': ram, 'cpu_pct': cpu,
            'completion_slo_attainment': attainment('completion_slo_met'),
            'ttft_slo_attainment': attainment('ttft_slo_met'),
            'tpot_slo_attainment': attainment('tpot_slo_met'),
            'overall_slo_attainment': attainment('overall_slo_met')})
    append_rows(OUT_DIR / 'summary.csv', SUMMARY_FIELDS, summaries)
    for s in summaries:
        print(f"  {s['group']:<14} {s['success']}/{s['requests']} ok, median e2e "
              f"{s['median_e2e_s'] or 0:.1f}s, p95 {s['p95_e2e_s'] or 0:.1f}s, "
              f"median TTFT {s['median_ttft_s'] or 0:.1f}s, {s['req_s']:.3f} req/s, "
              f"SLO overall {s['overall_slo_attainment']:.1%}")
    print(f'Appended results to {OUT_DIR}/requests.csv and summary.csv (run_id={labels["run_id"]})')


def evaluate_sim(results, reqs, nodes, args, run_id):
    """Summary rows per policy (all requests, each modality) from [(policy, log, decisions)], appended to sim_summary.csv and printed."""
    stem = Path(args.trace).stem
    price, has_price = cluster_price(nodes)
    summaries = []
    print(f'  {"policy":<7}{"group":<7}{"n":>5}{"p50 s":>9}{"p95 s":>9}{"SLO met":>9}'
          f'{"req/s":>8}{"$/1k":>7}{"decide µs":>11}  share per node')
    for policy, log, decisions in results:
        makespan = max(e['finish'] for e in log.values())
        decide_us = sum(decisions) / len(decisions) * 1e6
        groups = {'all': reqs, **{m: [r for r in reqs if r.modality == m] for m in MODALITIES}}
        for group, members in groups.items():
            if not members:
                continue
            e2e = [log[r.id]['finish'] - r.arrival for r in members]
            met = sum(log[r.id]['finish'] <= r.deadline for r in members) / len(members)
            usd = price * makespan / 3600 / len(reqs) * 1000 if group == 'all' and has_price else None
            share = ' '.join(f'{n.split(".")[0]}={sum(log[r.id]["node"] == n for r in members) / len(members):.0%}'
                             for n in nodes)
            summaries.append({
                'run_id': run_id, 'timestamp_utc': utc_now(),
                'trace': stem, 'policy': policy, 'seed': args.seed, 'slo_mult': args.slo,
                'group': group, 'requests': len(members), 'median_e2e_s': median(e2e),
                'p95_e2e_s': percentile(e2e, .95), 'slo_attainment': met,
                'req_s': len(members) / makespan if group == 'all' else None,
                'usd_per_1k_requests': usd, 'decision_us': decide_us if group == 'all' else None})
            print(f'  {policy if group == "all" else "":<7}{group:<7}{len(members):>5}'
                  f'{median(e2e):>9.1f}{percentile(e2e, .95):>9.1f}{met:>9.0%}'
                  + (f'{len(members) / makespan:>8.3f}{usd or 0:>7.2f}{decide_us:>11.1f}'
                     if group == 'all' else ' ' * 26) + f'  {share}')
    append_rows(OUT_DIR / 'sim_summary.csv', SIM_FIELDS, summaries)
    print(f'Appended results to {OUT_DIR}/sim_summary.csv (run_id={run_id})')


def main():
    parser = argparse.ArgumentParser(description='Open-loop load generator and router for the routing benchmarks.')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('capacity', help='print predicted per-node costs and cluster capacity')
    p.add_argument('--nodes', default='nodes.json')
    p.add_argument('--mix', default=DEFAULT_MIX)
    p = sub.add_parser('run', help='replay a trace over the nodes with one policy')
    p.add_argument(
        "--adaptive-config",
        default="config/adaptive.json",
        help="Adaptive calibration cache path",
    )

    p.add_argument(
        "--max-drain-s",
        type=float,
        default=1200.0,
        help="Maximum time to wait after all arrivals",
    )
    p.add_argument('--trace', required=True)
    p.add_argument('--nodes', default='nodes.json')
    p.add_argument('--policy', choices=(*POLICIES, "adaptive"), required=True)
    p.add_argument('--cluster', required=True, help='label for this set of nodes, e.g. A, B, C')
    p.add_argument('--seed', type=int, default=0, help='random policy and tie-breaks')
    p.add_argument('--output', choices=('natural', 'fixed'), default='natural')
    p.add_argument('--requests', type=int, help='replay only the first N requests (quick test)')
    p.add_argument('--no-warmup', action='store_true')
    p = sub.add_parser('sim', help='simulate a trace over the nodes with every policy')
    p.add_argument('--trace', required=True)
    p.add_argument('--nodes', default='nodes.json')
    p.add_argument('--policies', default='rr,least,ect,slo',
                   help='comma-separated, from ' + ', '.join(POLICIES + QUEUE_POLICIES))
    p.add_argument('--slots', default='c7i.xlarge=1',
                   help='slots per node as name=k,... (default 2 for nodes not listed)')
    p.add_argument('--slo', type=float, default=5.0,
                   help='SLO as a multiple of the alone time on the fastest node (default 5)')
    p.add_argument('--seed', type=int, default=0, help='service time draws and tie-breaks')
    p.add_argument('--requests', type=int, help='simulate only the first N requests')
    args = parser.parse_args()
    args.run_id = new_run_id()
    setup_logging(logging.DEBUG, policy=getattr(args, 'policy', args.command), run_id=args.run_id)
    {'capacity': capacity, 'run': run, 'sim': simulate}[args.command](args)


if __name__ == '__main__':
    main()
