
import pytest

from router.predictor import LatencyPredictor
from router.models import Request
from router.calibrate import fit_image_encoding, fit_audio_encoding, fit_token_stages, calibrate_node


@pytest.fixture
def mm_request():
    return Request(
        id=1,
        run="06",
        modality="image",
        prompt_tokens=1500,
        arrival=0,
        deadline=60,
    )

@pytest.fixture
def rows():
    rows = [
        # Image calibration
        {
            "modality": "image",
            "size": 1,
            "encoding_ms": 3200,
            "prefill_ms": 5000,
            "server_predicted_ms": 4000,
            "prompt_tokens": 1000,
            "output_tokens": 100,
        },
        {
            "modality": "image",
            "size": 4,
            "encoding_ms": 12800,
            "prefill_ms": 5000,
            "server_predicted_ms": 4000,
            "prompt_tokens": 1000,
            "output_tokens": 100,
        },

        # Audio calibration
        {
            "modality": "audio",
            "size": 10,
            "encoding_ms": 2200,
            "prefill_ms": 5000,
            "server_predicted_ms": 4000,
            "prompt_tokens": 1000,
            "output_tokens": 100,
        },
        {
            "modality": "audio",
            "size": 40,
            "encoding_ms": 8200,
            "prefill_ms": 5000,
            "server_predicted_ms": 4000,
            "prompt_tokens": 1000,
            "output_tokens": 100,
        },
    ]
    return rows


@pytest.fixture
def predictor(rows):
    stage_params = calibrate_node(rows)
    return LatencyPredictor(
        costs={"amd": {"06": 25.0}},
        model={
            "amd": {
                "a": {"image": 2.0},
                "b": {"image": 0.01},
                "d": 0.04,
                "out": {"06": 128},
            }
        },
        stage_params={"amd": stage_params}
    )


def test_predict_service(predictor, mm_request):
    expected = 2.0 + 0.01 * 1500 + 0.04 * 128

    predicted = predictor.predict_service(mm_request, "amd")

    assert predicted == pytest.approx(expected)


def test_median_fallback(predictor, mm_request):
    predicted = predictor.predict_service(
        mm_request, "amd", method="median"
    )

    assert predicted == pytest.approx(25.0)


def test_predict_stages(predictor, mm_request):


    result = predictor.predict_stages(
        mm_request,
        "amd",
        image_count=4,
        expected_output_tokens=100,
    )

    assert result.encoding_s == pytest.approx(12.8)
    assert result.prefill_s == pytest.approx(7.5)
    assert result.decode_s == pytest.approx(4.0)
    assert result.total_s == pytest.approx(24.3)



def test_fit_image_encoding():
    rows = [
        {"modality": "image", "size": 1, "encoding_ms": 3100},
        {"modality": "image", "size": 1, "encoding_ms": 3200},
        {"modality": "image", "size": 1, "encoding_ms": 3300},
        {"modality": "image", "size": 4, "encoding_ms": 12500},
        {"modality": "image", "size": 4, "encoding_ms": 12800},
        {"modality": "image", "size": 4, "encoding_ms": 13100},
    ]

    params = fit_image_encoding(rows)

    assert params["image_encode_base"] == pytest.approx(0.0)
    assert params["image_encode_per_image"] == pytest.approx(3.2)



def test_fit_audio_encoding():
    rows = [
        {"modality": "audio", "size": 10, "encoding_ms": 2000},
        {"modality": "audio", "size": 10, "encoding_ms": 2200},
        {"modality": "audio", "size": 10, "encoding_ms": 2400},
        {"modality": "audio", "size": 40, "encoding_ms": 8000},
        {"modality": "audio", "size": 40, "encoding_ms": 8200},
        {"modality": "audio", "size": 40, "encoding_ms": 8400},
    ]

    params = fit_audio_encoding(rows)

    assert params["audio_encode_base"] == pytest.approx(0.2)
    assert params["audio_encode_per_second"] == pytest.approx(0.2)


def test_fit_token_stages():
    rows = [
        {
            "prompt_tokens": 1000,
            "output_tokens": 100,
            "prefill_ms": 5000,
            "server_predicted_ms": 4000,
        },
        {
            "prompt_tokens": 2000,
            "output_tokens": 100,
            "prefill_ms": 10000,
            "server_predicted_ms": 4000,
        },
        {
            "prompt_tokens": 3000,
            "output_tokens": 100,
            "prefill_ms": 15000,
            "server_predicted_ms": 4000,
        },
    ]

    params = fit_token_stages(rows)

    assert params["prefill_s_per_token"] == pytest.approx(0.005)
    assert params["decode_s_per_token"] == pytest.approx(0.04)


def test_calibrate_node(rows):
    params = calibrate_node(rows)

    assert params["image_encode_base"] == pytest.approx(0.0)
    assert params["image_encode_per_image"] == pytest.approx(3.2)

    assert params["audio_encode_base"] == pytest.approx(0.2)
    assert params["audio_encode_per_second"] == pytest.approx(0.2)

    assert params["prefill_s_per_token"] == pytest.approx(0.005)
    assert params["decode_s_per_token"] == pytest.approx(0.04)


def test_stage_specific_online_update(predictor):
    observation = {
        "modality": "image",
        "image_count": 4,
        "prompt_tokens": 1000,
        "output_tokens": 100,
        "encoding_ms": 18000,
        "prompt_ms": 30000,
        "decode_ms": 6000,
    }

    slowdowns = {
        "encoding": 1.5,
        "prefill": 2.0,
        "decode": 1.5,
    }

    alpha = 0.1
    params = predictor.stage_params["amd"]

    # Save parameters before updating
    old_image_cost = params["image_encode_per_image"]
    old_prefill_cost = params["prefill_s_per_token"]
    old_decode_cost = params["decode_s_per_token"]

    # Convert observed times from ms to s
    encoding_s = observation["encoding_ms"] / 1000
    prefill_s = (
        observation["prompt_ms"] - observation["encoding_ms"]
    ) / 1000
    decode_s = observation["decode_ms"] / 1000

    # Calculate interference-normalized unit costs
    observed_image_cost = (
        encoding_s / slowdowns["encoding"]
        - params["image_encode_base"]
    ) / observation["image_count"]

    observed_prefill_cost = (
        prefill_s / slowdowns["prefill"]
    ) / observation["prompt_tokens"]

    observed_decode_cost = (
        decode_s / slowdowns["decode"]
    ) / observation["output_tokens"]

    # Expected EWMA updates
    expected_image_cost = (
        (1 - alpha) * old_image_cost
        + alpha * observed_image_cost
    )

    expected_prefill_cost = (
        (1 - alpha) * old_prefill_cost
        + alpha * observed_prefill_cost
    )

    expected_decode_cost = (
        (1 - alpha) * old_decode_cost
        + alpha * observed_decode_cost
    )

    # Execute update
    updated = predictor.update(
        "amd",
        observation,
        slowdowns=slowdowns,
        alpha=alpha,
    )

    assert updated is True
    assert params["image_encode_per_image"] == pytest.approx(
        expected_image_cost
    )
    assert params["prefill_s_per_token"] == pytest.approx(
        expected_prefill_cost
    )
    assert params["decode_s_per_token"] == pytest.approx(
        expected_decode_cost
    )

