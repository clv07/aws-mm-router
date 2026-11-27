import pytest
from router.models import Request
from router.router import Router
from router.models import StagePrediction


class FakePredictor:
    def predict_stages(
        self, request, node, image_count=0,
        audio_duration_s=0.0, expected_output_tokens=128,
    ):
        return StagePrediction(
            encoding_s=2.0,
            prefill_s=3.0,
            decode_s=expected_output_tokens * 0.1,
        )


class FakeInterference:
    def predict(self, node, victim, running_requests):
        factor = 1.5 if running_requests else 1.0
        return {
            "encoding": factor,
            "prefill": factor,
            "decode": factor,
        }


@pytest.fixture
def router():
    # Test these methods without Phase 1 CSV initialization.
    r = Router.__new__(Router)

    r.predictor = FakePredictor()
    r.interference = FakeInterference()

    r.output_estimates = {}
    r.default_output_tokens = 128

    r.names = ["amd"]
    r.slots = {"amd": 2}
    r.healthy = {"amd": True}
    r.mem_limit = {"amd": set()}
    r.active = {"amd": {}}

    r.request_slos = {
        1: {
            "ttft_s": 15.0,
            "tpot_s": 0.2,
        }
    }

    return r


def make_request(req_id):
    return Request(
        id=req_id,
        run="04",
        modality="image",
        prompt_tokens=1000,
        arrival=0.0,
        deadline=100.0,
    )


def test_cold_start_output_tokens(router):
    request = make_request(1)

    assert router.estimate_output_tokens(request) == 128


def test_output_token_ewma(router):
    request = make_request(1)

    router.update_output_tokens(request, 100)
    assert router.estimate_output_tokens(request) == 100

    router.update_output_tokens(request, 120, alpha=0.1)
    assert router.estimate_output_tokens(request) == pytest.approx(102)


def test_stage_cost_uses_shared_output_tokens(router):
    request = make_request(1)

    router.output_estimates["04"] = 100

    result = router.stage_cost(request, "amd", [])

    assert result["expected_output_tokens"] == 100
    assert result["decode_s"] == pytest.approx(10.0)
    assert result["ttft_s"] == pytest.approx(5.0)
    assert result["tpot_s"] == pytest.approx(0.1)
    assert result["total_s"] == pytest.approx(15.0)


def test_interference_affects_stage_cost(router):
    request = make_request(1)
    other = make_request(2)

    result = router.stage_cost(request, "amd", [other])

    assert result["slowdowns"]["encoding"] == 1.5
    assert result["encoding_s"] == pytest.approx(3.0)
    assert result["prefill_s"] == pytest.approx(4.5)
    assert result["decode_s"] == pytest.approx(19.2)


def test_candidate_includes_queue_wait(router):
    request = make_request(1)

    result = router.adaptive_candidate(
        request, "amd", now=12.0
    )

    # 12 seconds in queue + 5 seconds stage TTFT.
    assert result["ttft_s"] == pytest.approx(17.0)
    assert result["ttft_ok"] is False


def test_full_node_is_unavailable(router):
    router.active["amd"] = {
        2: {"request": make_request(2)},
        3: {"request": make_request(3)},
    }

    result = router.adaptive_candidate(
        make_request(1), "amd", now=0.0
    )

    assert result is None
