from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ClassificationResult:
    strategy_name: str
    answer: str
    confidence: float | None = None
    latency: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class JevDecision:
    answer: str
    confidence: float
    probabilities: dict[str, float]
    latency: float
    model: str
    usage: dict[str, int]
    answer_confidence: float | None = None
    routing: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMDecision:
    answer: str
    latency: float
    model: str
    usage: dict[str, int]

