"""Step 5.1/5.2: measured stage observations and request-level overlap."""

import pytest

import main
from router.interference import InterferencePredictor
from router.models import Request
from router.router import Router


def make_request(request_id, run, modality, arrival=0.0):
    return Request(
        id=request_id,
        run=run,
        modality=modality,
        prompt_tokens=1000.0,
        arrival=arrival,
        deadline=100.0,
    )


@pytest.fixture
def overlap_router():
    nodes = {"amd": {"slots": 2}, "intel": {"slots": 1}}
    params = {
        node: {
            "image_encode_base": 0.1,
            "image_encode_per_image": 0.2,
            "audio_encode_base": 0.1,
            "audio_encode_per_second": 0.01,
            "prefill_s_per_token": 0.001,
            "decode_s_per_token": 0.02,
        }
        for node in nodes
    }
    router = Router(
        policy="adaptive", nodes=nodes,
        costs={node: {} for node in nodes},
        gains={node: 1.0 for node in nodes},
        seed=0, stage_params=params,
        interference_predictor=InterferencePredictor(),
    )
    # Keep completion tests independent of the candidate evaluator.
    router.pick_adaptive = lambda now: None
    return router


def reserve(router, request, node):
    router.active[node][request.id] = {
        "request": request,
        "dispatch_s": request.arrival,
        "predicted_finish": 40.0,
        "http_start_s": None,
    }
    router.running[node][request.id] = 40.0


def test_image_stage_observation():
    request = make_request(1, "06", "image")
    result = {
        "status": "ok", "encoding_ms": 12000,
        "server_prompt_ms": 18000, "server_predicted_ms": 5000,
        "prompt_tokens": 1500, "output_tokens": 100,
    }
    obs = main.build_stage_observation(request, result)
    assert obs is not None
    assert obs["image_count"] == 4
    assert obs["audio_duration_s"] == 0.0
    assert obs["encoding_s"] == pytest.approx(12.0)
    assert obs["prefill_s"] == pytest.approx(6.0)
    assert obs["decode_s"] == pytest.approx(5.0)
    assert obs["prefill_ms"] == pytest.approx(6000.0)


def test_stage_observation_rejects_missing_timing():
    request = make_request(1, "06", "image")
    result = {
        "status": "ok", "encoding_ms": None,
        "server_prompt_ms": 18000, "server_predicted_ms": 5000,
        "prompt_tokens": 1500, "output_tokens": 100,
    }
    assert main.build_stage_observation(request, result) is None


def test_stage_observation_rejects_failure():
    request = make_request(1, "06", "image")
    assert main.build_stage_observation(
        request, {"status": "error", "error": "timeout"}
    ) is None


def test_stage_observation_rejects_invalid_prompt():
    request = make_request(1, "06", "image")
    result = {
        "status": "ok", "encoding_ms": 20000,
        "server_prompt_ms": 18000, "server_predicted_ms": 5000,
        "prompt_tokens": 1500, "output_tokens": 100,
    }
    assert main.build_stage_observation(request, result) is None


def test_completed_peer_still_counts_for_overlap(overlap_router):
    router = overlap_router
    a = make_request(1, "06", "image", 0)
    b = make_request(2, "07", "audio", 5)
    reserve(router, a, "amd")
    reserve(router, b, "amd")
    router.mark_http_start(a, "amd", 0)
    router.mark_http_start(b, "amd", 5)

    router.dispatch_adaptive(now=20, finished=(a, "amd", 20, True))
    overlap_a = router.pop_completed_overlap(a.id)
    assert overlap_a == [{
        "request_id": 2, "modality": "audio", "fraction": pytest.approx(0.75)
    }]
    assert a.id not in router.active["amd"]
    assert len(router.completed_intervals["amd"]) == 1

    router.dispatch_adaptive(now=25, finished=(b, "amd", 20, True))
    overlap_b = router.pop_completed_overlap(b.id)
    assert overlap_b == [{
        "request_id": 1, "modality": "image", "fraction": pytest.approx(0.75)
    }]
    assert router.active["amd"] == {}
    assert router.running["amd"] == {}
    assert router.pop_completed_overlap(b.id) == []


def test_overlap_is_node_local(overlap_router):
    router = overlap_router
    a = make_request(1, "06", "image", 0)
    b = make_request(2, "07", "audio", 0)
    reserve(router, a, "amd")
    reserve(router, b, "intel")
    router.mark_http_start(a, "amd", 0)
    router.mark_http_start(b, "intel", 0)
    router.dispatch_adaptive(now=10, finished=(a, "amd", 10, True))
    router.dispatch_adaptive(now=15, finished=(b, "intel", 15, True))
    assert router.pop_completed_overlap(a.id) == []
    assert router.pop_completed_overlap(b.id) == []


def test_duplicate_http_start_rejected(overlap_router):
    router = overlap_router
    request = make_request(1, "00", "chat", 0)
    reserve(router, request, "amd")
    router.mark_http_start(request, "amd", 0)
    with pytest.raises(RuntimeError, match="Duplicate HTTP start"):
        router.mark_http_start(request, "amd", 1)


def test_http_start_cannot_precede_dispatch(overlap_router):
    router = overlap_router
    request = make_request(1, "00", "chat", 2)
    reserve(router, request, "amd")
    with pytest.raises(ValueError, match="precedes dispatch"):
        router.mark_http_start(request, "amd", 1)

# Step 5.3: actual completion feedback drives online updates.
from router.models import StagePrediction


def stage_obs(modality="image", encoding=2.0, prefill=4.0, decode=8.0,
              prompt_tokens=2000, output_tokens=40):
    return {
        "modality": modality,
        "image_count": 4 if modality == "image" else 0,
        "audio_duration_s": 10.0 if modality == "audio" else 0.0,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "encoding_ms": encoding * 1000,
        "prompt_ms": (encoding + prefill) * 1000,
        "decode_ms": decode * 1000,
        "encoding_s": encoding,
        "prefill_s": prefill,
        "decode_s": decode,
    }


def test_solo_updates_stage_params_and_output_tokens(overlap_router):
    router = overlap_router
    request = make_request(11, "06", "image")
    router.output_estimates[request.run] = 20.0
    previous = dict(router.stage_params["amd"])
    feedback = router.apply_online_feedback(
        request=request, node="amd",
        observation=stage_obs(), overlaps=[],
        base_prediction=StagePrediction(1.0, 1.0, 0.4),
        expected_output_tokens=20,
        actual_output_tokens=40,
    )
    assert feedback == {
        "base_updated": True,
        "interference_updated": False,
        "output_updated": True,
    }
    assert router.stage_params["amd"]["prefill_s_per_token"] != previous["prefill_s_per_token"]
    assert router.stage_params["amd"]["decode_s_per_token"] != previous["decode_s_per_token"]
    assert router.output_estimates[request.run] == pytest.approx(22.0)
    assert router.interference.weights == {}


def test_overlap_updates_interference_not_base(overlap_router):
    router = overlap_router
    request = make_request(12, "06", "image")
    router.output_estimates[request.run] = 20.0
    initial = dict(router.stage_params["amd"])
    result = router.apply_online_feedback(
        request=request, node="amd",
        observation=stage_obs(),
        overlaps=[{"modality": "audio", "fraction": 0.75}],
        base_prediction=StagePrediction(1.0, 2.0, 2.0),
        expected_output_tokens=20,
        actual_output_tokens=40,
    )
    assert result["interference_updated"]
    assert not result["base_updated"]
    assert router.stage_params["amd"] == initial
    assert ("amd", "image", "audio", "decode") in router.interference.weights


def test_concurrent_decode_baseline_uses_actual_output_tokens(overlap_router):
    router = overlap_router
    request = make_request(13, "06", "image")
    # Predicted decode=2s for 20 tokens. Actual 4s for 40 tokens is NOT slowdown.
    result = router.apply_online_feedback(
        request=request, node="amd",
        observation=stage_obs(encoding=1, prefill=4, decode=4),
        overlaps=[{"modality": "audio", "fraction": 1.0}],
        base_prediction=StagePrediction(1.0, 2.0, 2.0),
        expected_output_tokens=20,
        actual_output_tokens=40,
    )
    assert result["interference_updated"]
    # A prior weight of 0.25 decreases when observed slowdown is exactly 1.
    assert router.interference.weights[("amd", "image", "audio", "decode")] < 0.25
    assert router.interference.weights[("amd", "image", "audio", "prefill")] < 0.25


def test_missing_stage_timing_can_still_update_output_estimate(overlap_router):
    router = overlap_router
    request = make_request(14, "01", "text")
    feedback = router.apply_online_feedback(
        request, "amd", observation=None, overlaps=[],
        base_prediction=None, expected_output_tokens=20,
        actual_output_tokens=35,
    )
    assert feedback == {
        "base_updated": False,
        "interference_updated": False,
        "output_updated": True,
    }
    assert router.output_estimates["01"] == pytest.approx(35)


def test_invalid_output_tokens_do_not_train(overlap_router):
    router = overlap_router
    request = make_request(15, "06", "image")
    feedback = router.apply_online_feedback(
        request, "amd", observation=None, overlaps=[],
        base_prediction=None, expected_output_tokens=20,
        actual_output_tokens=float("nan"),
    )
    assert not any(feedback.values())
    assert request.run not in router.output_estimates
