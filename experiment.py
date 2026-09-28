from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Iterable

from models import ClassificationResult
from strategies.base import ClassificationStrategy


DISPLAY_NAMES = {
    "raw-jev": "Raw Jev",
    "llm-only": "LLM Only",
}


@dataclass(frozen=True, slots=True)
class TestCase:
    id: int
    text: str
    choices: dict[str, str]
    expected_answer: str


class ExperimentRunner:
    def __init__(
        self,
        strategies: dict[str, ClassificationStrategy],
        output_path: Path,
        decision_backend: str | None = None,
    ) -> None:
        self._strategies = strategies
        self._output_path = output_path
        self._decision_backend = decision_backend

    def run(self, user_text: str, choices: dict[str, str]) -> list[ClassificationResult]:
        results = [
            strategy.classify(user_text, choices) for strategy in self._strategies.values()
        ]
        self._save(user_text, choices, results)
        print_interactive_results(user_text, results)
        return results

    def run_batch(self, cases: Iterable[TestCase]) -> list[tuple[TestCase, list[ClassificationResult]]]:
        runs: list[tuple[TestCase, list[ClassificationResult]]] = []
        for case in cases:
            results = [
                strategy.classify(case.text, case.choices)
                for strategy in self._strategies.values()
            ]
            self._save(case.text, case.choices, results, case)
            runs.append((case, results))
            print_case(case, results)
        print_batch_summary(runs)
        return runs

    def _save(
        self,
        user_text: str,
        choices: dict[str, str],
        results: list[ClassificationResult],
        case: TestCase | None = None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": user_text,
            "choices": choices,
            "case_id": case.id if case else None,
            "expected_answer": case.expected_answer if case else None,
            "decision_backend": self._decision_backend,
            "results": [result.to_dict() for result in results],
        }
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_cases(path: Path) -> list[TestCase]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Test cases must be a JSON array.")
        cases: list[TestCase] = []
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Each test case must be a JSON object.")
            raw_choices = item["choices"]
            if not isinstance(raw_choices, dict) or not raw_choices:
                raise ValueError("Each test case must define at least one choice.")
            if not all(
                isinstance(choice_id, str)
                and choice_id
                and isinstance(description, str)
                and description
                for choice_id, description in raw_choices.items()
            ):
                raise ValueError("Choice IDs and descriptions must be non-empty strings.")
            choices = dict(raw_choices)
            expected_answer = item["expected_answer"]
            if not isinstance(expected_answer, str) or expected_answer not in choices:
                raise ValueError("expected_answer must be one of the case's choice IDs.")
            cases.append(
                TestCase(
                    id=int(item["id"]),
                    text=str(item["text"]),
                    choices=choices,
                    expected_answer=expected_answer,
                )
            )
        return cases
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Could not load valid test cases from {path}.") from exc


def print_interactive_results(user_text: str, results: list[ClassificationResult]) -> None:
    print(f"\nOriginal input:\n{user_text}")
    for result in results:
        print_result(result)
    if len(results) > 1:
        print_agreement(results)


def print_result(result: ClassificationResult) -> None:
    print(f"\n{DISPLAY_NAMES[result.strategy_name]}:")
    print(f"Answer: {result.answer}")
    if result.confidence is not None:
        print(f"Confidence: {result.confidence:.3f}")
    if probabilities := result.details.get("probabilities"):
        print(f"Probabilities: {json.dumps(probabilities, ensure_ascii=False)}")
    if "llm_latency" in result.details and "decision_latency" in result.details:
        print(f"LLM latency: {result.details['llm_latency']:.3f}s")
        print(f"Decision latency: {result.details['decision_latency']:.3f}s")
    print(f"Total latency: {result.latency:.3f}s")


def print_agreement(results: list[ClassificationResult]) -> None:
    by_name = {result.strategy_name: result for result in results}
    pairs = [
        ("raw-jev", "llm-only"),
    ]
    print("\nAgreement:")
    for left, right in pairs:
        if left in by_name and right in by_name:
            verdict = "SAME" if by_name[left].answer == by_name[right].answer else "DIFFERENT"
            print(f"{DISPLAY_NAMES[left]} vs {DISPLAY_NAMES[right]}: {verdict}")


def print_case(case: TestCase, results: list[ClassificationResult]) -> None:
    print(f"\nCase {case.id}\nExpected: {case.expected_answer}")
    for result in results:
        print(f"{DISPLAY_NAMES[result.strategy_name]}: {result.answer} ({result.latency:.3f}s)")
def print_batch_summary(runs: list[tuple[TestCase, list[ClassificationResult]]]) -> None:
    grouped: dict[str, list[tuple[TestCase, ClassificationResult]]] = {}
    for case, results in runs:
        for result in results:
            grouped.setdefault(result.strategy_name, []).append((case, result))

    print("\nSummary")
    for name, items in grouped.items():
        accuracy = sum(result.answer == case.expected_answer for case, result in items) / len(items)
        print(f"{DISPLAY_NAMES[name]} accuracy: {accuracy:.1%}")

    print("\nAverage latency:")
    for name, items in grouped.items():
        print(f"{DISPLAY_NAMES[name]}: {fmean(result.latency for _, result in items):.3f}s")

    confidence_groups = {
        name: [result.confidence for _, result in items if result.confidence is not None]
        for name, items in grouped.items()
    }
    if any(confidence_groups.values()):
        print("\nAverage confidence:")
        for name, values in confidence_groups.items():
            if values:
                print(f"{DISPLAY_NAMES[name]}: {fmean(values):.3f}")


