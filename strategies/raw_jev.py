from __future__ import annotations

from clients.jev_client import DecisionClient
from models import ClassificationResult
from strategies.base import ClassificationStrategy


class RawJevStrategy(ClassificationStrategy):
    def __init__(self, decision_client: DecisionClient) -> None:
        self._decision = decision_client

    def classify(self, user_text: str, choices: dict[str, str]) -> ClassificationResult:
        decision = self._decision.classify(user_text, choices)
        return ClassificationResult(
            strategy_name="raw-jev",
            answer=decision.answer,
            confidence=decision.confidence,
            latency=decision.latency,
            details={
                "probabilities": decision.probabilities,
                "answer_confidence": decision.answer_confidence,
                "decision_latency": decision.latency,
                "decision_model": decision.model,
                "routing": decision.routing,
                "usage": decision.usage,
            },
        )

