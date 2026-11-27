STAGES = ("encoding", "prefill", "decode")


class InterferencePredictor:
    def __init__(
            self,
            initial_weights=None,
            unknown_slowdown=1.25,
            max_slowdown=8
    ):
        self.prior = unknown_slowdown - 1
        self.max_slowdown = max_slowdown

        self.weights = dict(initial_weights or {})

    def _key(self, node, victim, other, stage):
        return (
            node,
            victim["modality"],
            other["modality"],
            stage
        )

    def predict(self, node, victim, running_requests):
        """
            Predict stage-specific slowdown for a new request.

            victim / running_requests: {"modality": "image"}
        """
        result = {}

        for stage in STAGES:
            if (stage == "encoding" and victim["modality"] not in ("image", "audio")):
                result[stage] = 1.0
                continue

            excess = 0.0

            for other in running_requests:
                key = self._key(node, victim, other, stage)
                excess += self.weights.get(key, self.prior)

            result[stage] = min(
                1.0 + excess,
                self.max_slowdown
            )

        return result

    def update(
        self,
        node,
        victim,
        overlaps,
        observation,
        base_prediction,
        alpha=0.1
    ):
        """
        Learn interference from a completed request.

        overlaps:
            [
                {"modality": "audio", "fraction": 0.8},
            ]

        observation (seconds):
            {
                "encoding_s": 15.0,
                "prefill_s": 6.0,
                "decode_s": 5.0,
            }
        """
        exposure = {}

        for other in overlaps:
            modality = other["modality"]
            fraction = float(other["fraction"])

            exposure[modality] = (
                exposure.get(modality, 0.0) + fraction
            )

        # no interference
        if not exposure or not any(exposure.values()):
            return False

        updated = False

        stage_fields = {
            "encoding": "encoding_s",
            "prefill": "prefill_s",
            "decode": "decode_s",
        }

        for stage, field in stage_fields.items():
            if (
                stage == "encoding"
                and victim["modality"] not in ("image", "audio")
            ):
                continue

            actual = observation.get(field)
            base = getattr(base_prediction, field)

            if actual is None:
                continue

            actual = float(actual)
            base = float(base)

            observed_factor = actual / base
            observed_excess = max(0.0, min(observed_factor, self.max_slowdown) - 1.0)

            features = {}
            for modality, fraction in exposure.items():
                key = (
                    node,
                    victim["modality"],
                    modality,
                    stage
                )
                features[key] = fraction

            predicted_excess = sum(
                fraction * self.weights.get(key, self.prior)
                for key, fraction in features.items()
            )

            error = observed_excess - predicted_excess

            # nomalize gradient update for multiple inference
            denominoator = sum(
                fraction ** 2
                for fraction in features.values()
            )

            if denominoator == 0:
                continue

            for key, fraction in features.items():
                old_w = self.weights.get(key, self.prior)

                new_w = (
                    old_w + alpha * error * fraction / denominoator
                )

                self.weights[key] = max(
                    0.0,
                    min(new_w, self.max_slowdown - 1.0)
                )

            updated = True
        return updated


