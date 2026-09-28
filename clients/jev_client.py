from __future__ import annotations

from time import perf_counter
from typing import Any, Protocol

import requests

from models import JevDecision


class SystemOneClientError(RuntimeError):
    pass


class DecisionClient(Protocol):
    def classify(self, state: str | dict[str, Any], choices: dict[str, str]) -> JevDecision: ...


class SystemOneClient:
    def __init__(
        self,
        api_key: str | None,
        model: str | None,
        base_url: str,
        timeout: float,
        service_name: str = "Decision backend",
        discover_model: bool = False,
        session: requests.Session | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._timeout = timeout
        self._service_name = service_name
        self._discover_model_enabled = discover_model
        self._session = session or requests.Session()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._session.headers.update(headers)

    def classify(self, state: str | dict[str, Any], choices: dict[str, str]) -> JevDecision:
        if not choices:
            raise ValueError("At least one choice is required.")
        payload = {
            "state": state,
            "questions": {
                "answer": {
                    "type": "choice",
                    "instructions": "Choose the best answer to the question from the supplied choices.",
                    "criteria": dict(choices),
                }
            },
        }
        model = self._model
        if model is None and self._discover_model_enabled:
            model = self._discover_model()
        if model is not None:
            payload["model"] = model

        started = perf_counter()
        data = self._request("POST", "/v1/systemone", json=payload)
        latency = perf_counter() - started
        try:
            answer = data["answers"]["answer"]
            answer_id = str(answer["choice"])
            # Laya's `confidence` is normalized-entropy concentration. Its
            # `answer_confidence` is max(probabilities); preserve both fields.
            confidence = float(answer["confidence"])
            probabilities = {str(k): float(v) for k, v in answer["probabilities"].items()}
            raw_answer_confidence = answer.get("answer_confidence")
            answer_confidence = (
                float(raw_answer_confidence) if raw_answer_confidence is not None else None
            )
            raw_routing = data.get("routing")
            routing = dict(raw_routing) if isinstance(raw_routing, dict) else {}
            response_model = str(routing.get("model") or data.get("model") or model or "auto")
            usage = {str(k): int(v) for k, v in data["usage"].items()}
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise SystemOneClientError(
                f"{self._service_name} returned an unexpected response shape."
            ) from exc

        if answer.get("type") != "choice" or answer_id not in choices:
            raise SystemOneClientError(
                f"{self._service_name} returned an answer outside the supplied choices."
            )
        return JevDecision(
            answer_id,
            confidence,
            probabilities,
            latency,
            response_model,
            usage,
            answer_confidence,
            routing,
        )

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def _discover_model(self) -> str:
        data = self._request("GET", "/v1/models")
        try:
            models = data["models"]
            model = str(models[0]["name"])
        except (KeyError, IndexError, TypeError) as exc:
            raise SystemOneClientError("TypeSafe Jev did not return an available model.") from exc
        self._model = model
        return model

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._session.request(
                method, f"{self._base_url}{path}", timeout=self._timeout, **kwargs
            )
            response.raise_for_status()
            data = response.json()
        except requests.Timeout as exc:
            message = f"{self._service_name} request timed out."
            if self._service_name == "Laya":
                message += " The server may still be downloading or loading model weights."
            raise SystemOneClientError(message) from exc
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            reason = _error_reason(exc.response)
            suffix = f": {reason}" if reason else ""
            raise SystemOneClientError(
                f"{self._service_name} request failed with HTTP {status}{suffix}."
            ) from exc
        except requests.RequestException as exc:
            if self._service_name == "Laya":
                raise SystemOneClientError(
                    "Cannot connect to local Laya server. Start it with: laya-serve"
                ) from exc
            raise SystemOneClientError("Could not reach the TypeSafe Jev API.") from exc
        except ValueError as exc:
            raise SystemOneClientError(f"{self._service_name} returned invalid JSON.") from exc
        if not isinstance(data, dict):
            raise SystemOneClientError(
                f"{self._service_name} returned an unexpected response shape."
            )
        return data


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


# Backwards-compatible names for callers that imported the original client.
JevClient = SystemOneClient
JevClientError = SystemOneClientError

