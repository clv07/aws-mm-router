
import threading

import pytest

from router.models import Request, StagePrediction
from router.router import Router


def make_request(req_id, deadline=100.0):
    return Request(
        id=req_id,
        run="04",
        modality="image",
        prompt_tokens=1000,
        arrival=0.0,
        deadline=deadline,
    )


@pytest.fixture
def router():
    # Isolate the scheduler from CSV-based initialization.
    r = Router.__new__(Router)

    r.policy = "adaptive"
    r.names = ["amd", "intel"]
    r.slots = {"amd": 1, "intel": 1}
    r.lock = threading.Lock()
    r.queue = []
    r.active = {"amd": {}, "intel": {}}
    r.running = {"amd": {}, "intel": {}}
    r.healthy = {"amd": True, "intel": True}

    def candidate(request, node, now):
        if len(r.active[node]) >= r.slots[node]:
            return None

        duration = 10.0 if node == "amd" else 15.0

        return {
            "node": node,
            "predicted": {
                "base": StagePrediction(1.0, 1.0, duration - 2),
                "expected_output_tokens": 100,
            },
            "queue_wait_s": now - request.arrival,
            "completion_s": now + duration,
            "violations": 0,
            "added_delay_s": 0.0,
        }

    # Stub only the candidate evaluator.
    r.adaptive_candidate = candidate
    return r


def test_pick_least_slack_request(router):
    less_urgent = make_request(1, deadline=100)
    more_urgent = make_request(2, deadline=30)

    router.queue = [less_urgent, more_urgent]

    request, candidate = router.pick_adaptive(now=0.0)

    assert request.id == 2
    assert candidate["node"] == "amd"


def test_dispatch_reserves_node_slots(router):
    first = make_request(1)
    second = make_request(2)

    router.queue = [first, second]

    dispatched = router.dispatch_adaptive(now=0.0)

    assert len(dispatched) == 2
    assert len(router.active["amd"]) == 1
    assert len(router.active["intel"]) == 1
    assert router.queue == []


def test_wait_and_reschedule_after_completion(router):
    first = make_request(1)
    second = make_request(2)
    waiting = make_request(3)

    router.dispatch_adaptive(now=0.0, arrived=first)
    router.dispatch_adaptive(now=0.0, arrived=second)

    result = router.dispatch_adaptive(
        now=1.0, arrived=waiting
    )

    assert result == []
    assert waiting in router.queue

    result = router.dispatch_adaptive(
        now=10.0,
        finished=(first, "amd", 10.0, True),
    )

    assert result == [(waiting, "amd")]
    assert waiting not in router.queue
    assert waiting.id in router.active["amd"]
    assert first.id not in router.active["amd"]
    assert first.id not in router.running["amd"]
