from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clients.jev_client import SystemOneClient, SystemOneClientError
from config import Settings
from strategies.raw_jev import RawJevStrategy


PROJECT_DIR = Path(__file__).resolve().parent
LABELS = ("A", "B", "C", "D")
DEFAULT_SEED = 20260928


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the independent Laya option-position-bias experiment."
    )
    parser.add_argument("--cases", default="cases.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def load_and_validate_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 100:
        raise ValueError("The experiment requires exactly 100 cases.")

    ids: list[int] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each case must be an object.")
        case_id = item.get("id")
        text = item.get("text")
        choices = item.get("choices")
        expected = item.get("expected_answer")
        if not isinstance(case_id, int):
            raise ValueError("Every case ID must be an integer.")
        if not isinstance(text, str) or not text:
            raise ValueError(f"Case {case_id} has invalid question text.")
        if not isinstance(choices, dict) or tuple(choices) != LABELS:
            raise ValueError(f"Case {case_id} must have ordered A/B/C/D choices.")
        if not all(isinstance(value, str) and value for value in choices.values()):
            raise ValueError(f"Case {case_id} has an invalid choice value.")
        if expected not in choices:
            raise ValueError(f"Case {case_id} has an invalid expected answer.")
        ids.append(case_id)

    if len(ids) != len(set(ids)):
        raise ValueError("Case IDs must be unique.")
    return data


def build_work_items(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for case in cases:
        original_choices = case["choices"]
        for rotation in range(4):
            displayed_choices: dict[str, str] = {}
            displayed_to_original: dict[str, str] = {}
            original_to_displayed: dict[str, str] = {}
            for original_index, original_label in enumerate(LABELS):
                displayed_label = LABELS[(original_index + rotation) % len(LABELS)]
                displayed_choices[displayed_label] = original_choices[original_label]
                displayed_to_original[displayed_label] = original_label
                original_to_displayed[original_label] = displayed_label

            # Restore canonical displayed order after assigning the rotated contents.
            displayed_choices = {
                label: displayed_choices[label]
                for label in LABELS
            }
            displayed_to_original = {
                label: displayed_to_original[label]
                for label in LABELS
            }
            items.append(
                {
                    "original_case_id": case["id"],
                    "question_text": case["text"],
                    "original_choices": original_choices,
                    "original_expected_answer": case["expected_answer"],
                    "permutation_version": f"rotation_{rotation}",
                    "rotation": rotation,
                    "original_a_position": original_to_displayed["A"],
                    "choices": displayed_choices,
                    "displayed_to_original": displayed_to_original,
                    "original_to_displayed": original_to_displayed,
                    "expected_display_answer": original_to_displayed[
                        case["expected_answer"]
                    ],
                }
            )
    return items


def warm_up(strategy: RawJevStrategy) -> list[dict[str, Any]]:
    questions = [
        (
            "Which animal is a mammal?",
            {"A": "Shark", "B": "Dolphin", "C": "Trout", "D": "Octopus"},
        ),
        ("What is seven plus five?", {"A": "10", "B": "11", "C": "12", "D": "13"}),
        (
            "If all roses are flowers and this is a rose, what is it?",
            {"A": "A flower", "B": "A mineral", "C": "A planet", "D": "A number"},
        ),
    ]
    warmups: list[dict[str, Any]] = []
    for index, (text, choices) in enumerate(questions, 1):
        result = strategy.classify(text, choices)
        model = result.details.get("decision_model")
        routing = result.details.get("routing", {})
        if model != "english" or routing.get("model") != "english":
            raise RuntimeError(f"Warm-up {index} did not route to the English checkpoint.")
        warmups.append(
            {
                "warmup": index,
                "answer": result.answer,
                "latency_ms": result.latency * 1000,
                "model": model,
            }
        )
    return warmups


def success_record(
    item: dict[str, Any], request_index: int, seed: int, result: Any
) -> dict[str, Any]:
    predicted_original = item["displayed_to_original"][result.answer]
    probabilities = result.details.get("probabilities", {})
    probabilities_by_original = {
        item["displayed_to_original"][displayed]: probability
        for displayed, probability in probabilities.items()
    }
    return {
        "experiment": "laya_option_position_bias_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_index": request_index,
        "shuffle_seed": seed,
        "original_case_id": item["original_case_id"],
        "permutation_version": item["permutation_version"],
        "rotation": item["rotation"],
        "original_a_position": item["original_a_position"],
        "question_text": item["question_text"],
        "choices": item["choices"],
        "displayed_to_original": item["displayed_to_original"],
        "original_to_displayed": item["original_to_displayed"],
        "original_expected_answer": item["original_expected_answer"],
        "expected_display_answer": item["expected_display_answer"],
        "decision_backend": "laya",
        "strategy_name": result.strategy_name,
        "status": "success",
        "prediction": result.answer,
        "predicted_original_option": predicted_original,
        "predicted_original_option_text": item["original_choices"][predicted_original],
        "is_correct": predicted_original == item["original_expected_answer"],
        "latency_s": result.latency,
        "confidence": result.confidence,
        "answer_confidence": result.details.get("answer_confidence"),
        "probabilities": probabilities,
        "probabilities_by_original_option": probabilities_by_original,
        "decision_model": result.details.get("decision_model"),
        "routing": result.details.get("routing", {}),
        "usage": result.details.get("usage", {}),
    }


def failure_record(
    item: dict[str, Any], request_index: int, seed: int, error: Exception
) -> dict[str, Any]:
    return {
        "experiment": "laya_option_position_bias_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_index": request_index,
        "shuffle_seed": seed,
        "original_case_id": item["original_case_id"],
        "permutation_version": item["permutation_version"],
        "rotation": item["rotation"],
        "original_a_position": item["original_a_position"],
        "question_text": item["question_text"],
        "choices": item["choices"],
        "displayed_to_original": item["displayed_to_original"],
        "original_to_displayed": item["original_to_displayed"],
        "original_expected_answer": item["original_expected_answer"],
        "expected_display_answer": item["expected_display_answer"],
        "decision_backend": "laya",
        "strategy_name": "raw-jev",
        "status": "failure",
        "error_type": type(error).__name__,
        "error": str(error),
    }


def main() -> int:
    args = parse_args()
    case_path = Path(args.cases)
    output_path = Path(args.output)
    if not case_path.is_absolute():
        case_path = PROJECT_DIR / case_path
    if not output_path.is_absolute():
        output_path = PROJECT_DIR / output_path
    if output_path.exists():
        raise FileExistsError(f"Refusing to append to existing output: {output_path}")

    settings = Settings.load()
    if settings.decision_backend != "laya":
        raise RuntimeError("DECISION_BACKEND must be laya for this experiment.")
    settings.validate_decision_base_url()
    decision = SystemOneClient(
        api_key=settings.laya_api_key,
        model=None,
        base_url=settings.laya_base_url,
        timeout=settings.request_timeout,
        service_name="Laya",
    )
    health = decision.health()
    if health.get("status") != "ok" or health.get("device") != "cuda":
        raise RuntimeError(f"Laya CUDA health check failed: {health}")
    if "english" not in health.get("loaded", []):
        raise RuntimeError("The Laya English checkpoint is not loaded.")

    strategy = RawJevStrategy(decision)
    warmups = warm_up(strategy)
    print(json.dumps({"health": health, "warmups": warmups}, ensure_ascii=False), flush=True)

    cases = load_and_validate_cases(case_path)
    items = build_work_items(cases)
    if len(items) != 400:
        raise RuntimeError(f"Expected 400 work items, got {len(items)}.")
    random.Random(args.seed).shuffle(items)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success_count = 0
    failure_count = 0
    with output_path.open("x", encoding="utf-8") as output:
        for request_index, item in enumerate(items, 1):
            try:
                result = strategy.classify(item["question_text"], item["choices"])
                record = success_record(item, request_index, args.seed, result)
                success_count += 1
            except (SystemOneClientError, ValueError) as error:
                record = failure_record(item, request_index, args.seed, error)
                failure_count += 1
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            if request_index % 25 == 0 or request_index == len(items):
                print(
                    f"progress={request_index}/400 success={success_count} failures={failure_count}",
                    flush=True,
                )

    print(f"output={output_path}", flush=True)
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
