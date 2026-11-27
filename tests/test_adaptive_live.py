import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

import main
from benchmark import RUNS


def fake_nodes():
    return {
        "amd": {
            "url": "http://localhost:8080",
            "model": "test-model",
            "slots": 1,
        }
    }


def fake_params():
    return {
        "image_encode_base": 0.1,
        "image_encode_per_image": 0.2,
        "audio_encode_base": 0.1,
        "audio_encode_per_second": 0.01,
        "prefill_s_per_token": 0.001,
        "decode_s_per_token": 0.02,
    }


def fake_cache(nodes):
    return {
        "signature": main.calibration_signature(
            nodes, "natural"
        ),
        "stage_params": {
            "amd": fake_params()
        },
        "prompt_tokens_by_run": {
            run: 100.0 for run in RUNS
        },
        "output_tokens_by_run": {
            run: 20.0 for run in RUNS
        },
    }


def test_cache_hit(monkeypatch, tmp_path):
    nodes = fake_nodes()
    path = tmp_path / "adaptive.json"

    expected = fake_cache(nodes)

    path.write_text(
        json.dumps(expected),
        encoding="utf-8",
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Calibration ran on cache hit"
        )

    monkeypatch.setattr(
        main,
        "collect_calibration_samples",
        fail_if_called,
    )

    result = main.load_or_calibrate(
        nodes,
        path,
        "natural",
    )

    assert result == expected


def test_cache_miss(monkeypatch, tmp_path):
    nodes = fake_nodes()
    path = tmp_path / "adaptive.json"

    samples = {
        "amd": [
            {
                "run": run,
                "modality": modality,
                "size": size,
                "prompt_tokens": 100.0,
                "output_tokens": 20.0,
            }
            for run, (modality, size) in RUNS.items()
        ]
    }

    monkeypatch.setattr(
        main,
        "load_datasets",
        lambda modalities: {},
    )

    monkeypatch.setattr(
        main,
        "collect_calibration_samples",
        lambda *args: samples,
    )

    monkeypatch.setattr(
        main,
        "calibrate_node",
        lambda rows: fake_params(),
    )

    result = main.load_or_calibrate(
        nodes,
        path,
        "natural",
    )

    assert path.exists()
    assert result["stage_params"]["amd"] == fake_params()
    assert len(result["prompt_tokens_by_run"]) == 9

    # Second call should read the saved cache.
    reloaded = main.load_or_calibrate(
        nodes,
        path,
        "natural",
    )

    assert reloaded == result


def test_calibration_failure_does_not_save_cache(
    monkeypatch, tmp_path
):
    nodes = fake_nodes()
    path = tmp_path / "adaptive.json"

    monkeypatch.setattr(
        main,
        "load_datasets",
        lambda modalities: {},
    )

    def fail(*args):
        raise RuntimeError("Missing encoding_ms")

    monkeypatch.setattr(
        main,
        "collect_calibration_samples",
        fail,
    )

    with pytest.raises(
        RuntimeError,
        match="Missing encoding_ms",
    ):
        main.load_or_calibrate(
            nodes, path, "natural"
        )

    assert not path.exists()


class FakeRouter:
    def __init__(self):
        self.lock = threading.Lock()
        self.slots = {"amd": 1}
        self.active = {"amd": {}}
        self.queue = []
        self.request_slos = {}
        self.prompt_tokens_by_run = {
            run: 100.0 for run in RUNS
        }

        self.dispatch_log = []
        self.completion_log = []

    def dispatch_adaptive(
        self, now, arrived=None, finished=None
    ):
        with self.lock:
            if arrived is not None:
                self.queue.append(arrived)

            if finished is not None:
                request, node, actual_s, success = finished
                self.active[node].pop(request.id)
                self.completion_log.append(request.id)

            dispatched = []

            while self.queue and not self.active["amd"]:
                request = self.queue.pop(0)

                self.active["amd"][request.id] = {
                    "request": request,
                    "dispatch_s": now,
                    "predicted_finish": now + 1.0,
                }

                self.dispatch_log.append(request.id)
                dispatched.append((request, "amd"))

            return dispatched


def test_wait_and_reschedule(monkeypatch, tmp_path):
    nodes = fake_nodes()
    router = FakeRouter()

    allow_a_finish = threading.Event()
    a_started = threading.Event()
    b_started = threading.Event()

    trace = [
        {
            "id": "1",
            "arrival_s": "0",
            "run": "00",
            "modality": "chat",
            "size": "short",
            "input": "1",
            "deadline_s": "30",
            "ttft_deadline_s": "10",
            "tpot_slo_ms": "200",
        },
        {
            "id": "2",
            "arrival_s": "0.01",
            "run": "00",
            "modality": "chat",
            "size": "short",
            "input": "2",
            "deadline_s": "30",
            "ttft_deadline_s": "10",
            "tpot_slo_ms": "200",
        },
    ]

    bodies = {
        "1": {"request_id": 1},
        "2": {"request_id": 2},
    }

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class FakeMeasurements:
        def __init__(self, *args):
            self.thread = threading.Thread(
                target=lambda: None
            )

        def record(self, *args):
            pass

        def complete(self, *args):
            pass

        def close(self):
            self.thread.join()

    def fake_request_once(session, url, body):
        if body["request_id"] == 1:
            a_started.set()
            if not allow_a_finish.wait(timeout=3):
                raise TimeoutError("A was never released")
        else:
            b_started.set()

        return {
            "status": "ok",
            "error": "",
            "finish_reason": "stop",
            "e2e_s": 0.01,
            "ttft_s": 0.005,
            "decode_tok_s": 20.0,
            "prompt_tokens": 100,
            "output_tokens": 20,
            "server_prompt_ms": 4.0,
            "server_predicted_ms": 5.0,
        }

    monkeypatch.setattr(
        main.requests,
        "Session",
        FakeSession,
    )

    monkeypatch.setattr(
        main,
        "request_once",
        fake_request_once,
    )

    monkeypatch.setattr(
        main,
        "Measurements",
        FakeMeasurements,
    )

    monkeypatch.setattr(
        main,
        "append_rows",
        lambda *args: None,
    )

    monkeypatch.setattr(
        main,
        "OUT_DIR",
        tmp_path,
    )

    args = SimpleNamespace(max_drain_s=3.0)

    labels = {
        "run_id": "test",
        "cluster": "test",
        "trace": "test",
        "policy": "adaptive",
        "seed": 0,
        "output_mode": "natural",
    }

    result_holder = {}
    errors = []

    def run_replay():
        try:
            result_holder["result"] = (
                main.replay_adaptive(
                    args,
                    nodes,
                    trace,
                    router,
                    bodies,
                    labels,
                )
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_replay)
    thread.start()

    try:
        assert a_started.wait(timeout=2)

        # Wait for B to arrive and enter the queue.
        deadline = __import__("time").monotonic() + 2

        while __import__("time").monotonic() < deadline:
            with router.lock:
                queued = len(router.queue)
            if queued == 1:
                break
            threading.Event().wait(0.005)

        with router.lock:
            assert len(router.queue) == 1
            assert len(router.active["amd"]) == 1

        # B must not start while A owns the only slot.
        assert not b_started.is_set()

    finally:
        allow_a_finish.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert not errors, errors
    assert b_started.is_set()
    assert router.dispatch_log == [1, 2]
    assert router.completion_log == [1, 2]

    rows, makespan, monitors = result_holder["result"]

    assert len(rows) == 2
    assert all(r["status"] == "ok" for r in rows)
    assert makespan > 0
    assert router.active["amd"] == {}
    assert router.queue == []