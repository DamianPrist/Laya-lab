from __future__ import annotations

from abc import ABC, abstractmethod

from models import ClassificationResult


class ClassificationStrategy(ABC):
    @abstractmethod
    def classify(self, user_text: str, choices: dict[str, str]) -> ClassificationResult:
        raise NotImplementedError

