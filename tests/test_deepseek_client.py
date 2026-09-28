from __future__ import annotations

import json
import unittest
from typing import Any

import requests

from clients.deepseek_client import DeepSeekClient, DeepSeekClientError


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
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def deepseek_response(answer: str, model: str = "deepseek-flash") -> dict[str, Any]:
    return {
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps({"answer": answer})}],
            }
        ],
        "usage": {"input_tokens": 25, "output_tokens": 5, "total_tokens": 30},
    }


class DeepSeekClientTests(unittest.TestCase):
    def test_each_request_constrains_schema_to_only_its_current_choices(self) -> None:
        session = FakeSession(
            [FakeResponse(deepseek_response("A")), FakeResponse(deepseek_response("RED"))]
        )
        client = DeepSeekClient(
            api_key="secret",
            model="deepseek-flash",
            base_url="https://api.deepseek.com",
            timeout=30,
            session=session,  # type: ignore[arg-type]
        )
        first = {"A": "one", "B": "two"}
        second = {"RED": "red", "GREEN": "green", "BLUE": "blue"}

        first_result = client.classify("Pick a number", first)
        second_result = client.classify("Pick a color", second)

        self.assertEqual(first_result.answer, "A")
        self.assertEqual(second_result.answer, "RED")
        self.assertEqual(
            first_result.usage,
            {"input_tokens": 25, "output_tokens": 5, "total_tokens": 30},
        )
        schemas = [call[1]["json"]["text"]["format"]["schema"] for call in session.calls]
        self.assertEqual(schemas[0]["properties"]["answer"]["enum"], list(first))
        self.assertEqual(schemas[1]["properties"]["answer"]["enum"], list(second))
        payloads = [json.loads(call[1]["json"]["input"]) for call in session.calls]
        self.assertEqual(payloads[0], {"question": "Pick a number", "choices": first})
        self.assertEqual(payloads[1], {"question": "Pick a color", "choices": second})
        self.assertEqual(session.calls[0][0], "https://api.deepseek.com/responses")

    def test_rejects_answer_outside_current_choices(self) -> None:
        session = FakeSession([FakeResponse(deepseek_response("C"))])
        client = DeepSeekClient(
            api_key="secret",
            model="deepseek-flash",
            base_url="https://api.deepseek.com",
            timeout=30,
            session=session,  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(DeepSeekClientError, "outside the supplied choices"):
            client.classify("Pick one", {"A": "one", "B": "two"})


if __name__ == "__main__":
    unittest.main()
