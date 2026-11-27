from .models import Request, StagePrediction



class LatencyPredictor:
    def __init__(self, costs, model, stage_params=None):
        self.costs = costs
        self.model = model
        self.stage_params = stage_params

    def predict_service(self, request, node, method="token"):
        fit = self.model[node] if self.model else {}
        if (
            method == "token"
            and request.modality in fit.get("a", {})
            and request.modality in fit.get("b", {})
            and request.run in fit.get("out", {})
        ):
            return (
                fit["a"][request.modality]
                + fit["b"][request.modality] * request.prompt_tokens
                + fit["d"] * fit["out"][request.run]
            )
        return self.costs[node][request.run]

    def predict_stages(
            self,
            request: Request,
            node: str,
            image_count:int = 0,
            audio_duration_s: float = 0,
            expected_output_tokens: float = 128
    ):
        params = self.stage_params[node]

        if request.modality == "image":
            encoding_s = (
                params["image_encode_base"]
                + params["image_encode_per_image"] * image_count
            )
        elif request.modality == "audio":
            encoding_s = (
                params["audio_encode_base"]
                + params["audio_encode_per_second"] * audio_duration_s
            )
        else:
            encoding_s = 0.0

        prefill_s = (
            params["prefill_s_per_token"] * request.prompt_tokens
        )

        decode_s = (
            expected_output_tokens * params["decode_s_per_token"]
        )

        return StagePrediction(
            encoding_s=encoding_s,
            prefill_s=prefill_s,
            decode_s=decode_s
        )


    def update(self, node: str, observation: dict, slowdowns:float=None, alpha:float=0.1):
        """
        Update stage-specific base costs using EWMA.

        observation:
            modality: "image", "audio", or "chat"
            image_count: int (image requests)
            audio_duration_s: float (audio requests)
            prompt_tokens: int
            output_tokens: int
            encoding_ms: float
            prefill_ms: float
            decode_ms: float

        slowdowns:
            {
                "encoding": float,
                "prefill": float,
                "decode": float,
            }

        A missing slowdown means the corresponding stage
        will not update its base cost.
        """
        params = self.stage_params[node]
        slowdowns = slowdowns if slowdowns is not None else {}
        updated = False

        def get_slowdown(stage):
            factor = slowdowns.get(stage)

            if factor is None:
                return None

            factor = float(factor)

            return factor

        def ewma(param_key, observed_cost):
            nonlocal updated

            old_cost = params.get(param_key)

            if old_cost is None:
                params[param_key] = observed_cost
            else:
                params[param_key] = (1 - alpha) * old_cost + alpha * observed_cost

            updated = True

        modality = observation.get("modality")
        encoding_ms = observation.get("encoding_ms")
        prompt_ms = observation.get("prompt_ms")
        decode_ms = observation.get("decode_ms")
        encoding_slowdown = get_slowdown("encoding")

        encoding_s = (
            encoding_ms / 1000
            if encoding_ms is not None else None
        )

        prefill_s = (
            (prompt_ms - encoding_ms) / 1000
            if prompt_ms is not None and encoding_ms is not None
            else None
        )

        decode_s = (
            decode_ms / 1000
            if decode_ms is not None else None
        )

        # Encode stage
        if encoding_s is not None and encoding_slowdown is not None:
            if modality == "image":
                image_count = observation.get("image_count", 0)

                if image_count > 0:
                    base = params["image_encode_base"]

                    observed_cost = (
                        encoding_s / encoding_slowdown - base
                    ) / image_count

                    ewma(
                        "image_encode_per_image",
                        max(0.0, observed_cost),
                    )
            elif modality == "audio":
                audio_duration_s = observation.get(
                    "audio_duration_s", 0
                )

                if audio_duration_s > 0:
                    base = params["audio_encode_base"]

                    observed_cost = (
                        encoding_s / encoding_slowdown - base
                    ) / audio_duration_s

                    ewma(
                        "audio_encode_per_second",
                        max(0.0, observed_cost),
                    )
        # Prefill Stage
        prompt_tokens = observation.get("prompt_tokens", 0)
        prefill_slowdown = get_slowdown("prefill")

        if (
            prefill_s is not None
            and prompt_tokens > 0
            and prefill_slowdown is not None
        ):
            observed_cost = (
                prefill_s / prompt_tokens / prefill_slowdown
            )

            ewma("prefill_s_per_token", observed_cost)

        # Decode stages
        output_tokens = observation.get("output_tokens", 0)
        decode_slowdown = get_slowdown("decode")

        if (
            decode_s is not None
            and output_tokens > 0
            and decode_slowdown is not None
        ):
            observed_cost = (
                decode_s / output_tokens / decode_slowdown
            )

            ewma("decode_s_per_token", observed_cost)

        return updated


            