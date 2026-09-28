from __future__ import annotations

from clients.deepseek_client import DeepSeekClient
from models import ClassificationResult
from strategies.base import ClassificationStrategy


class LLMOnlyStrategy(ClassificationStrategy):
    def __init__(self, llm_client: DeepSeekClient) -> None:
        self._llm = llm_client

    def classify(self, user_text: str, choices: dict[str, str]) -> ClassificationResult:
        decision = self._llm.classify(user_text, choices)
        return ClassificationResult(
            strategy_name="llm-only",
            answer=decision.answer,
            latency=decision.latency,
            details={
                "llm_latency": decision.latency,
                "llm_model": decision.model,
                "usage": decision.usage,
            },
        )

