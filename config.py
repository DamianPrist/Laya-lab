from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    decision_backend: str
    deepseek_api_key: str | None
    deepseek_model: str | None
    deepseek_base_url: str
    laya_api_key: str | None
    laya_base_url: str
    typesafe_api_key: str | None
    typesafe_model: str | None
    typesafe_base_url: str
    request_timeout: float

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        decision_backend = os.getenv("DECISION_BACKEND", "laya").strip().lower()
        if decision_backend not in {"laya", "jev"}:
            raise ValueError("DECISION_BACKEND must be either 'laya' or 'jev'.")
        return cls(
            decision_backend=decision_backend,
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
            deepseek_model=os.getenv("DEEPSEEK_MODEL") or None,
            deepseek_base_url=os.getenv(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
            ).rstrip("/"),
            laya_api_key=os.getenv("LAYA_API_KEY") or None,
            laya_base_url=os.getenv("LAYA_BASE_URL", "http://127.0.0.1:8000").rstrip("/"),
            typesafe_api_key=os.getenv("TYPESAFE_API_KEY") or None,
            typesafe_model=os.getenv("TYPESAFE_MODEL") or None,
            typesafe_base_url=os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/"),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
        )

    def missing_for(self, strategy: str) -> list[str]:
        missing: list[str] = []
        if strategy in {"llm-only", "compare"}:
            if not self.deepseek_api_key:
                missing.append("DEEPSEEK_API_KEY")
            if not self.deepseek_model:
                missing.append("DEEPSEEK_MODEL")
        if (
            self.decision_backend == "jev"
            and strategy in {"raw-jev", "compare"}
            and not self.typesafe_api_key
        ):
            missing.append("TYPESAFE_API_KEY")
        return missing

    @property
    def decision_backend_display_name(self) -> str:
        return "Laya" if self.decision_backend == "laya" else "TypeSafe Jev"

    @property
    def decision_base_url(self) -> str:
        return self.laya_base_url if self.decision_backend == "laya" else self.typesafe_base_url

    def validate_decision_base_url(self) -> None:
        parsed = urlparse(self.decision_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            variable = "LAYA_BASE_URL" if self.decision_backend == "laya" else "TYPESAFE_BASE_URL"
            raise ValueError(f"{variable} must be a valid HTTP(S) URL.")

