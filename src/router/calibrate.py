from statistics import median
from typing import List


def calibrate_node(rows):
    image_params = fit_image_encoding(rows)
    audio_params = fit_audio_encoding(rows)
    token_params = fit_token_stages(rows)

    return {
        **image_params,
        **audio_params,
        **token_params,
    }

def fit_image_encoding(rows:List[dict]):
    """
        Fit image encoding cost from request-level measurements
    """
    by_size = {}

    for row in rows:
        if row["modality"] != "image":
            continue

        size = int(row["size"])
        encoding_s = float(row["encoding_ms"]) / 1000

        by_size.setdefault(size, []).append(encoding_s)

    if len(by_size) < 2:
        raise ValueError("At leaset two images sizes are required")

    sizes = sorted(by_size)

    x1, x2 = sizes[0], sizes[-1]
    y1 = median(by_size[x1])
    y2 = median(by_size[x2])

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    return {
        "image_encode_base": intercept,
        "image_encode_per_image": slope
    }


def fit_audio_encoding(rows:List[dict]):
    by_duration = {}

    for row in rows:
        if row["modality"] != "audio":
            continue

        duration = float(row["size"])
        encoding_s = float(row["encoding_ms"]) / 1000

        by_duration.setdefault(duration, []).append(encoding_s)

    if len(by_duration) < 2:
        raise ValueError("At least two audio durations are required")

    durations = sorted(by_duration)

    x1, x2 = durations[0], durations[-1]
    y1 = median(by_duration[x1])
    y2 = median(by_duration[x2])

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    return {
        "audio_encode_base": intercept,
        "audio_encode_per_second": slope,
    }


def fit_token_stages(rows):
    """Fit prefill and decode costs from request measurements."""

    prefill_costs = []
    decode_costs = []

    for row in rows:
        prompt_tokens = float(row["prompt_tokens"])
        output_tokens = float(row["output_tokens"])

        if prompt_tokens > 0 and row.get("prefill_ms") is not None:
            prefill_costs.append(
                float(row["prefill_ms"]) / 1000 / prompt_tokens
            )

        # Decode timing already exists in the original benchmark.
        decode_ms = row.get("decode_ms", row.get("server_predicted_ms"))

        if output_tokens > 0 and decode_ms is not None:
            decode_costs.append(
                float(decode_ms) / 1000 / output_tokens
            )

    if not prefill_costs or not decode_costs:
        raise ValueError("Insufficient prefill or decode measurements")

    return {
        "prefill_s_per_token": median(prefill_costs),
        "decode_s_per_token": median(decode_costs),
    }
