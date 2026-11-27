from dataclasses import dataclass
from typing import Optional


@dataclass(eq=False)
class Request:
    id: int
    run: str
    modality: str
    prompt_tokens: float
    arrival: float
    deadline: float


@dataclass
class StagePrediction:
    encoding_s: float
    prefill_s: float
    decode_s: float

    @property
    def total_s(self) -> float:
        return (
            self.encoding_s
            + self.prefill_s
            + self.decode_s
        )


@dataclass
class Decision:
    action: str
    request_id: Optional[int] = None
    node: Optional[str] = None
