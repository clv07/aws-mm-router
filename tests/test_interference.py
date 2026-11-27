from router.interference import InterferencePredictor
from router.models import StagePrediction
import pytest

def test_cold_start_prediction():
    predictor = InterferencePredictor(
        unknown_slowdown=1.25
    )

    result = predictor.predict(
        node="amd",
        victim={"modality": "image"},
        running_requests=[
            {"modality": "audio"},
            {"modality": "chat"},
        ],
    )

    # 1 + (1.25 - 1) + (1.25 - 1) = 1.5
    for stage in ("encoding", "prefill", "decode"):
        assert result[stage] == pytest.approx(1.5)


def test_directional_interference():
    predictor = InterferencePredictor(
        initial_weights={
            ("amd", "image", "audio", "encoding"): 0.5,
            ("amd", "audio", "image", "encoding"): 1.0,
        },
    )

    image = predictor.predict(
        "amd",
        {"modality": "image"},
        [{"modality": "audio"}],
    )

    audio = predictor.predict(
        "amd",
        {"modality": "audio"},
        [{"modality": "image"}],
    )

    assert image["encoding"] == pytest.approx(1.5)
    assert audio["encoding"] == pytest.approx(2.0)


def test_online_update_without_csv():
    predictor = InterferencePredictor(
        unknown_slowdown=1.25
    )

    victim = {"modality": "image"}
    running = [{"modality": "audio"}]

    base = StagePrediction(
        encoding_s=10.0,
        prefill_s=5.0,
        decode_s=4.0,
    )

    observation = {
        "encoding_s": 15.0,
        "prefill_s": 6.0,
        "decode_s": 5.0,
    }

    before = predictor.predict("amd", victim, running)

    updated = predictor.update(
        node="amd",
        victim=victim,
        overlaps=[
            {"modality": "audio", "fraction": 1.0},
        ],
        observation=observation,
        base_prediction=base,
        alpha=0.5,
    )

    after = predictor.predict("amd", victim, running)

    # Encoding:
    # observed slowdown = 15 / 10 = 1.5
    # old weight = 1.25 - 1 = 0.25
    # observed excess = 1.5 - 1 = 0.5
    # new weight = 0.25 + 0.5 * (0.5 - 0.25)
    expected_encoding = 1 + (
        0.25 + 0.5 * (0.5 - 0.25)
    )

    # Prefill:
    # observed slowdown = 6 / 5 = 1.2
    # new weight = 0.25 + 0.5 * (0.2 - 0.25)
    expected_prefill = 1 + (
        0.25 + 0.5 * (0.2 - 0.25)
    )

    # Decode:
    # observed slowdown = 5 / 4 = 1.25
    # new weight = 0.25 (unchanged)
    expected_decode = 1.25

    assert updated is True

    assert before["encoding"] == pytest.approx(1.25)
    assert after["encoding"] == pytest.approx(expected_encoding)
    assert after["prefill"] == pytest.approx(expected_prefill)
    assert after["decode"] == pytest.approx(expected_decode)


def test_no_overlap_does_not_update():
    predictor = InterferencePredictor()

    base = StagePrediction(
        encoding_s=10.0,
        prefill_s=5.0,
        decode_s=4.0,
    )

    updated = predictor.update(
        node="amd",
        victim={"modality": "image"},
        overlaps=[],
        observation={"encoding_s": 15.0},
        base_prediction=base,
    )

    assert updated is False
    assert predictor.weights == {}


def test_multiple_running_requests():
    predictor = InterferencePredictor(
        initial_weights={
            ("amd", "image", "audio", "encoding"): 0.4,
        }
    )

    result = predictor.predict(
        node="amd",
        victim={"modality": "image"},
        running_requests=[
            {"modality": "audio"},
            {"modality": "audio"},
        ],
    )

    # 1 + 2 * 0.4 = 1.8
    assert result["encoding"] == pytest.approx(1.8)