from __future__ import annotations

import unittest
from typing import Any

import requests

from clients.jev_client import SystemOneClient, SystemOneClientError
from config import Settings
from main import build_strategies


class FakeResponse:
    def __init__(self, data: dict[str, Any], status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code
        self.reason = "test error" if status_code >= 400 else "OK"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError()
            error.response = self  # type: ignore[assignment]
            raise error

    def json(self) -> dict[str, Any]:
        return self._data


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.headers: dict[str, str] = {}
        self.responses = responses
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


def decision_response(answer: str = "A", model: str = "english") -> dict[str, Any]:
    return {
        "model": model,
        "answers": {
            "answer": {
                "type": "choice",
                "choice": answer,
                "confidence": 0.91,
                "probabilities": {answer: 0.91},
            }
        },
        "usage": {"input_tokens": 12, "output_tokens": 1},
    }


class SystemOneClientTests(unittest.TestCase):
    def test_laya_omits_model_and_authorization_when_unconfigured(self) -> None:
        session = FakeSession([FakeResponse(decision_response())])
        client = SystemOneClient(
            api_key=None,
            model=None,
            base_url="http://127.0.0.1:8000",
            timeout=30,
            service_name="Laya",
            session=session,  # type: ignore[arg-type]
        )

        choices = {"A": "refund", "B": "keep purchase"}
        result = client.classify("refund please", choices)

        self.assertEqual(result.answer, "A")
        self.assertNotIn("Authorization", session.headers)
        payload = session.calls[0][2]["json"]
        self.assertNotIn("model", payload)
        self.assertEqual(payload["questions"]["answer"]["criteria"], choices)
        self.assertEqual(session.calls[0][1], "http://127.0.0.1:8000/v1/systemone")

    def test_laya_preserves_both_confidences_and_routing(self) -> None:
        response = decision_response("B")
        response["answers"]["answer"].update(
            {
                "confidence": 0.064,
                "answer_confidence": 0.507,
                "probabilities": {"A": 0.2866, "B": 0.507, "C": 0.2064},
            }
        )
        response["routing"] = {
            "model": "multilingual",
            "reason": "non-Latin script",
        }
        session = FakeSession([FakeResponse(response)])
        client = SystemOneClient(
            api_key=None,
            model=None,
            base_url="http://127.0.0.1:8000",
            timeout=30,
            service_name="Laya",
            session=session,  # type: ignore[arg-type]
        )

        result = client.classify("问题", {"A": "一", "B": "二", "C": "三"})

        self.assertEqual(result.confidence, 0.064)
        self.assertEqual(result.answer_confidence, 0.507)
        self.assertEqual(result.probabilities["B"], 0.507)
        self.assertEqual(result.model, "multilingual")
        self.assertEqual(result.routing["reason"], "non-Latin script")

    def test_api_key_adds_bearer_header(self) -> None:
        session = FakeSession([])
        SystemOneClient(
            api_key="secret",
            model=None,
            base_url="http://127.0.0.1:8000",
            timeout=30,
            session=session,  # type: ignore[arg-type]
        )
        self.assertEqual(session.headers["Authorization"], "Bearer secret")

    def test_typesafe_model_discovery_is_preserved(self) -> None:
        session = FakeSession(
            [
                FakeResponse({"models": [{"name": "jev-test"}]}),
                FakeResponse(decision_response(model="jev-test")),
            ]
        )
        client = SystemOneClient(
            api_key="secret",
            model=None,
            base_url="https://api.typesafe.ai",
            timeout=30,
            service_name="TypeSafe Jev",
            discover_model=True,
            session=session,  # type: ignore[arg-type]
        )

        client.classify("refund please", {"A": "refund", "B": "other"})

        self.assertEqual(session.calls[0][1], "https://api.typesafe.ai/v1/models")
        self.assertEqual(session.calls[1][2]["json"]["model"], "jev-test")

    def test_http_error_keeps_status_and_safe_reason(self) -> None:
        session = FakeSession([FakeResponse({"detail": "invalid question"}, 422)])
        client = SystemOneClient(
            api_key=None,
            model=None,
            base_url="http://127.0.0.1:8000",
            timeout=30,
            service_name="Laya",
            session=session,  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(SystemOneClientError, "HTTP 422: invalid question"):
            client.classify("refund please", {"A": "refund", "B": "other"})

    def test_each_request_uses_its_own_choices(self) -> None:
        session = FakeSession(
            [
                FakeResponse(decision_response("B")),
                FakeResponse(decision_response("GREEN")),
            ]
        )
        client = SystemOneClient(
            api_key=None,
            model=None,
            base_url="http://127.0.0.1:8000",
            timeout=30,
            service_name="Laya",
            session=session,  # type: ignore[arg-type]
        )
        first = {"A": "3", "B": "4"}
        second = {"RED": "red", "GREEN": "green", "BLUE": "blue"}

        self.assertEqual(client.classify("2 + 2?", first).answer, "B")
        self.assertEqual(client.classify("Grass color?", second).answer, "GREEN")
        self.assertEqual(
            [call[2]["json"]["questions"]["answer"]["criteria"] for call in session.calls],
            [first, second],
        )


class SettingsTests(unittest.TestCase):
    def make_settings(self, backend: str, typesafe_api_key: str | None = None) -> Settings:
        return Settings(
            decision_backend=backend,
            deepseek_api_key=None,
            deepseek_model=None,
            deepseek_base_url="https://api.deepseek.com",
            laya_api_key=None,
            laya_base_url="http://127.0.0.1:8000",
            typesafe_api_key=typesafe_api_key,
            typesafe_model=None,
            typesafe_base_url="https://api.typesafe.ai",
            request_timeout=30,
        )

    def test_raw_laya_does_not_require_cloud_keys(self) -> None:
        self.assertEqual(self.make_settings("laya").missing_for("raw-jev"), [])

    def test_raw_jev_still_requires_typesafe_key(self) -> None:
        self.assertEqual(
            self.make_settings("jev").missing_for("raw-jev"), ["TYPESAFE_API_KEY"]
        )

    def test_compare_builds_only_raw_jev_and_llm_only(self) -> None:
        settings = self.make_settings("laya")
        settings = Settings(
            decision_backend=settings.decision_backend,
            deepseek_api_key="test-key",
            deepseek_model="deepseek-flash",
            deepseek_base_url=settings.deepseek_base_url,
            laya_api_key=settings.laya_api_key,
            laya_base_url=settings.laya_base_url,
            typesafe_api_key=settings.typesafe_api_key,
            typesafe_model=settings.typesafe_model,
            typesafe_base_url=settings.typesafe_base_url,
            request_timeout=settings.request_timeout,
        )
        self.assertEqual(list(build_strategies("compare", settings)), ["raw-jev", "llm-only"])


if __name__ == "__main__":
    unittest.main()
