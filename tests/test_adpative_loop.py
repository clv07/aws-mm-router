"""Step 5.4: end-to-end scheduler feedback tests, without real AWS HTTP."""

from concurrent.futures import ThreadPoolExecutor

import pytest

import main
from router.interference import InterferencePredictor
from router.models import Request
from router.router import Router


def make_router(slots=1):
    nodes = {"amd": {"slots": slots}, "intel": {"slots": slots}}
    params = {
        "amd": {
            "image_encode_base": 0.1,
            "image_encode_per_image": 0.2,
            "audio_encode_base": 0.1,
            "audio_encode_per_second": 0.01,
            "prefill_s_per_token": 0.001,
            "decode_s_per_token": 0.01,
        },
        "intel": {
            "image_encode_base": 0.1,
            "image_encode_per_image": 0.2,
            "audio_encode_base": 0.1,
            "audio_encode_per_second": 0.01,
            "prefill_s_per_token": 0.002,
            "decode_s_per_token": 0.012,
        },
    }
    router = Router(
        policy="adaptive", nodes=nodes,
        costs={name: {} for name in nodes},
        gains={name: 1.0 for name in nodes},
        seed=0, stage_params=params,
        interference_predictor=InterferencePredictor(),
    )
    router.output_estimates["01"] = 20.0
    return router


def req(request_id, arrival):
    return Request(
        id=request_id, run="01", modality="text", prompt_tokens=1000.0,
        arrival=arrival, deadline=arrival + 100.0,
    )


def set_slo(router, request):
    router.request_slos[request.id] = {"ttft_s": 100.0, "tpot_s": 100.0}


def text_observation(prefill_seconds, output_tokens=20, decode_seconds=0.2):
    return {
        "modality": "text", "image_count": 0, "audio_duration_s": 0,
        "prompt_tokens": 1000, "output_tokens": output_tokens,
        "encoding_ms": 0.0,
        "prompt_ms": prefill_seconds * 1000,
        "decode_ms": decode_seconds * 1000,
        "encoding_s": 0.0, "prefill_s": prefill_seconds,
        "decode_s": decode_seconds,
    }


def test_solo_feedback_changes_next_node_selection():
    router = make_router()
    a = req(1, 0)
    b = req(2, 30)
    set_slo(router, a)
    set_slo(router, b)

    before_amd = router.stage_cost(b, "amd", [])["total_s"]
    before_intel = router.stage_cost(b, "intel", [])["total_s"]
    assert before_amd < before_intel

    first = router.dispatch_adaptive(now=0, arrived=a)
    assert first == [(a, "amd")]
    snapshot = dict(router.active["amd"][a.id])
    router.mark_http_start(a, "amd", 0.0)
    assert router.dispatch_adaptive(now=20, finished=(a, "amd", 20, True)) == []
    overlaps = router.pop_completed_overlap(a.id)
    assert overlaps == []

    result = router.apply_online_feedback(
        request=a, node="amd", observation=text_observation(20),
        overlaps=overlaps, base_prediction=snapshot["base_prediction"],
        expected_output_tokens=snapshot["expected_output_tokens"],
        actual_output_tokens=20,
    )
    assert result["base_updated"]
    after_amd = router.stage_cost(b, "amd", [])["total_s"]
    after_intel = router.stage_cost(b, "intel", [])["total_s"]
    assert after_amd > before_amd
    assert after_amd > after_intel

    second = router.dispatch_adaptive(now=30, arrived=b)
    assert second == [(b, "intel")]


def test_output_feedback_changes_next_decode_prediction():
    router = make_router()
    a, b = req(1, 0), req(2, 20)
    prior = router.stage_cost(b, "amd", [])["decode_s"]
    result = router.apply_online_feedback(
        request=a, node="amd", observation=None,
        overlaps=[], base_prediction=None,
        expected_output_tokens=20, actual_output_tokens=100,
    )
    assert result == {
        "base_updated": False,
        "interference_updated": False,
        "output_updated": True,
    }
    assert router.output_estimates["01"] == pytest.approx(28.0)
    assert router.stage_cost(b, "amd", [])["decode_s"] == pytest.approx(prior * 1.4)


def test_failed_http_result_does_not_update_models():
    router = make_router()
    a = req(1, 0)
    before = dict(router.stage_params["amd"])
    weights_before = dict(router.interference.weights)
    observation = main.build_stage_observation(a, {"status": "error", "error": "HTTP 500"})
    assert observation is None
    # Worker contract: feedback is only invoked on successful responses.
    if observation is not None:
        router.apply_online_feedback(
            request=a, node="amd", observation=observation,
            overlaps=[], base_prediction=None,
            expected_output_tokens=20, actual_output_tokens=100,
        )
    assert router.stage_params["amd"] == before
    assert router.interference.weights == weights_before
    assert router.output_estimates["01"] == 20.0


def test_concurrent_feedback_updates_are_atomic():
    router = make_router()
    inputs = [req(i, 0) for i in range(1, 101)]

    def update(request):
        return router.apply_online_feedback(
            request=request, node="amd", observation=None,
            overlaps=[], base_prediction=None,
            expected_output_tokens=20, actual_output_tokens=100,
        )

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(update, inputs))

    assert all(result["output_updated"] for result in results)
    # Exact EWMA value after N serialized updates, regardless of thread order.
    expected = 100 - (100 - 20) * (0.9 ** len(inputs))
    assert router.output_estimates["01"] == pytest.approx(expected)


def test_overlap_feedback_changes_future_interference_prediction():
    router = make_router(slots=2)
    router.output_estimates["06"] = 20.0
    image = Request(id=10, run="06", modality="image", prompt_tokens=1000,
                    arrival=0.0, deadline=100)
    audio = Request(id=11, run="07", modality="audio", prompt_tokens=1000,
                    arrival=2.0, deadline=100)
    solo = router.stage_cost(image, "amd", [])
    predicted_concurrent = router.stage_cost(image, "amd", [audio])
    before = predicted_concurrent["total_s"]
    assert before > solo["total_s"]

    # Create actual overlapping HTTP intervals on the *same* node.
    for request, start in ((image, 0.0), (audio, 2.0)):
        router.active["amd"][request.id] = {
            "request": request, "dispatch_s": start,
            "predicted_finish": start + 15.0,
            "base_prediction": router.stage_cost(request, "amd", [])["base"],
            "expected_output_tokens": 20.0,
        }
        router.running["amd"][request.id] = start + 15.0
        router.mark_http_start(request, "amd", start)

    snapshot = dict(router.active["amd"][image.id])
    router.dispatch_adaptive(now=10.0, finished=(image, "amd", 10.0, True))
    overlaps = router.pop_completed_overlap(image.id)
    assert overlaps[0]["modality"] == "audio"
    assert overlaps[0]["fraction"] == pytest.approx(0.8)

    # More interference than the default prior predicts, at identical token lengths.
    base = snapshot["base_prediction"]
    observation = {
        "modality": "image", "image_count": 4, "audio_duration_s": 0,
        "prompt_tokens": 1000, "output_tokens": 20,
        "encoding_s": base.encoding_s * 2.5,
        "prefill_s": base.prefill_s * 2.5,
        "decode_s": base.decode_s * 2.5,
    }
    observation.update({
        "encoding_ms": observation["encoding_s"] * 1000,
        "prompt_ms": (observation["encoding_s"] + observation["prefill_s"]) * 1000,
        "decode_ms": observation["decode_s"] * 1000,
    })
    params_before = dict(router.stage_params["amd"])
    feedback = router.apply_online_feedback(
        request=image, node="amd", observation=observation,
        overlaps=overlaps, base_prediction=base,
        expected_output_tokens=20, actual_output_tokens=20,
    )
    assert feedback["interference_updated"]
    assert not feedback["base_updated"]
    assert router.stage_params["amd"] == params_before
    assert router.stage_cost(image, "amd", [audio])["total_s"] > before
