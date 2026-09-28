from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import requests

from models import LLMDecision


class DeepSeekClientError(RuntimeError):
    pass


CLASSIFICATION_INSTRUCTIONS = """Answer the supplied multiple-choice question.
Select exactly one supplied choice ID and return it in the required structured response."""


@dataclass(frozen=True, slots=True)
class StructuredResponse:
    value: dict[str, Any]
    latency: float
    model: str
    usage: dict[str, int]


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float,
        session: requests.Session | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def classify(self, user_text: str, choices: dict[str, str]) -> LLMDecision:
        if not choices:
            raise ValueError("At least one choice is required.")
        schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "enum": list(choices),
                    "description": "The selected choice ID.",
                }
            },
            "required": ["answer"],
            "additionalProperties": False,
        }
        response = self._request_structured(
            instructions=CLASSIFICATION_INSTRUCTIONS,
            user_input={"question": user_text, "choices": choices},
            schema_name="choice_answer",
            schema=schema,
            max_output_tokens=64,
        )
        answer = response.value.get("answer")
        if not isinstance(answer, str) or answer not in choices:
            raise DeepSeekClientError("DeepSeek returned an answer outside the supplied choices.")
        return LLMDecision(answer, response.latency, response.model, response.usage)

    def _request_structured(
        self,
        instructions: str,
        user_input: dict[str, object],
        schema_name: str,
        schema: dict[str, Any],
        max_output_tokens: int,
    ) -> StructuredResponse:
        payload = {
            "model": self._model,
            "instructions": instructions,
            "input": json.dumps(user_input, ensure_ascii=False),
            "reasoning": {"effort": "none"},
            "temperature": 0,
            "max_output_tokens": max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                }
            },
        }
        started = perf_counter()
        try:
            response = self._session.post(
                f"{self._base_url}/responses",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.Timeout as exc:
            raise DeepSeekClientError("DeepSeek request timed out.") from exc
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            reason = _error_reason(exc.response)
            suffix = f": {reason}" if reason else ""
            raise DeepSeekClientError(
                f"DeepSeek request failed with HTTP {status}{suffix}."
            ) from exc
        except requests.RequestException as exc:
            raise DeepSeekClientError("Could not reach the DeepSeek API.") from exc
        except ValueError as exc:
            raise DeepSeekClientError("DeepSeek returned invalid JSON.") from exc
        latency = perf_counter() - started

        try:
            if data["status"] != "completed":
                raise DeepSeekClientError(
                    f"DeepSeek response did not complete (status: {data['status']})."
                )
            output_text = next(
                part["text"]
                for item in data["output"]
                if item.get("type") == "message"
                for part in item.get("content", [])
                if part.get("type") == "output_text"
            )
            value = json.loads(output_text)
            if not isinstance(value, dict):
                raise TypeError
            response_model = str(data.get("model") or self._model)
            usage = _normalize_usage(data.get("usage"))
        except DeepSeekClientError:
            raise
        except (KeyError, StopIteration, TypeError, ValueError, AttributeError) as exc:
            raise DeepSeekClientError(
                "DeepSeek returned an unexpected structured response."
            ) from exc
        return StructuredResponse(value, latency, response_model, usage)


def _normalize_usage(raw_usage: object) -> dict[str, int]:
    if not isinstance(raw_usage, dict):
        return {}
    usage: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = raw_usage.get(key)
        if isinstance(value, int):
            usage[key] = value
    return usage


def _error_reason(response: requests.Response | None) -> str | None:
    if response is None:
        return None
    try:
        data = response.json()
    except ValueError:
        return response.reason or None
    if not isinstance(data, dict):
        return response.reason or None
    error = data.get("error")
    if isinstance(error, dict):
        message = error.get("message")
    else:
        message = data.get("detail") or data.get("message")
    return str(message)[:300] if message else (response.reason or None)
