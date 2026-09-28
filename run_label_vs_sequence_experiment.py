from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clients.jev_client import SystemOneClient, SystemOneClientError
from config import Settings
from strategies.raw_jev import RawJevStrategy


PROJECT_DIR = Path(__file__).resolve().parent
LABELS = ("A", "B", "C", "D")
DEFAULT_SEED = 20260928
SAMPLE_SIZE = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separate Laya label bias from candidate sequence-position bias."
    )
    parser.add_argument("--cases", default="cases.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary")
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


def make_item(
    case: dict[str, Any],
    sample_rank: int,
    control_group: str,
    variant: str,
    sequence: list[tuple[str, str]],
) -> dict[str, Any]:
    """Build criteria in sequence order without canonical-label re-sorting.

    Each tuple in sequence is (current_label, original_label). Insertion order is
    therefore the candidate sequence order that Laya receives.
    """
    original_choices = case["choices"]
    criteria: dict[str, str] = {}
    label_to_original: dict[str, str] = {}
    sequence_entries: list[dict[str, Any]] = []
    for position, (label, original_label) in enumerate(sequence, 1):
        if label in criteria:
            raise ValueError(f"Duplicate current label {label} in {variant}.")
        criteria[label] = original_choices[original_label]
        label_to_original[label] = original_label
        sequence_entries.append(
            {
                "position": position,
                "label": label,
                "original_option": original_label,
                "content": original_choices[original_label],
            }
        )

    original_to_label = {original: label for label, original in label_to_original.items()}
    return {
        "original_case_id": case["id"],
        "sample_rank": sample_rank,
        "question_text": case["text"],
        "original_choices": dict(original_choices),
        "original_expected_answer": case["expected_answer"],
        "control_group": control_group,
        "variant": variant,
        "criteria": criteria,
        "criteria_order": list(criteria),
        "sequence_entries": sequence_entries,
        "label_to_original": label_to_original,
        "original_to_label": original_to_label,
        "expected_current_label": original_to_label[case["expected_answer"]],
        "a_sequence_position": list(criteria).index("A") + 1,
        "first_sequence_label": next(iter(criteria)),
    }


def build_work_items(
    cases: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[int]]:
    sampled = random.Random(seed).sample(cases, SAMPLE_SIZE)
    sampled_ids = [case["id"] for case in sampled]
    items: list[dict[str, Any]] = []

    for sample_rank, case in enumerate(sampled, 1):
        baseline_sequence = [(label, label) for label in LABELS]
        baseline = make_item(
            case,
            sample_rank,
            "shared_baseline",
            "baseline",
            baseline_sequence,
        )
        baseline["analysis_groups"] = ["position_control", "label_control"]
        items.append(baseline)

        # Keep each label bound to its original content, and only rotate the
        # actual dict insertion order. The rotations place A at positions 2..4.
        for a_position in range(2, 5):
            shift = a_position - 1
            order = LABELS[-shift:] + LABELS[:-shift]
            item = make_item(
                case,
                sample_rank,
                "position_control",
                f"a_at_sequence_position_{a_position}",
                [(label, label) for label in order],
            )
            item["analysis_groups"] = ["position_control"]
            items.append(item)

        # Keep original contents in their A/B/C/D physical sequence positions,
        # and rotate only the literal labels assigned to those positions.
        for rotation in range(1, 4):
            sequence = [
                (LABELS[(position + rotation) % 4], original_label)
                for position, original_label in enumerate(LABELS)
            ]
            item = make_item(
                case,
                sample_rank,
                "label_control",
                f"first_sequence_label_{sequence[0][0]}",
                sequence,
            )
            item["analysis_groups"] = ["label_control"]
            items.append(item)

    validate_work_items(items, sampled_ids)
    random.Random(seed).shuffle(items)
    return items, sampled_ids


def validate_work_items(items: list[dict[str, Any]], sampled_ids: list[int]) -> None:
    if len(sampled_ids) != SAMPLE_SIZE or len(set(sampled_ids)) != SAMPLE_SIZE:
        raise RuntimeError("The fixed-seed sample must contain 20 distinct cases.")
    if len(items) != SAMPLE_SIZE * 7:
        raise RuntimeError(f"Expected 140 unique requests, got {len(items)}.")

    by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if tuple(item["criteria"]) != tuple(item["criteria_order"]):
            raise RuntimeError("criteria_order does not match actual dict insertion order.")
        if set(item["criteria"]) != set(LABELS):
            raise RuntimeError("Every request must contain labels A/B/C/D exactly once.")
        by_case[item["original_case_id"]].append(item)

    for case_id, case_items in by_case.items():
        baseline = [item for item in case_items if item["control_group"] == "shared_baseline"]
        if len(baseline) != 1:
            raise RuntimeError(f"Case {case_id} must have one shared baseline.")

        position_items = baseline + [
            item for item in case_items if item["control_group"] == "position_control"
        ]
        if sorted(item["a_sequence_position"] for item in position_items) != [1, 2, 3, 4]:
            raise RuntimeError(f"Case {case_id} does not place A in all four positions.")
        for item in position_items:
            if item["label_to_original"] != {label: label for label in LABELS}:
                raise RuntimeError(f"Case {case_id} position control changed label bindings.")

        label_items = baseline + [
            item for item in case_items if item["control_group"] == "label_control"
        ]
        if sorted(item["first_sequence_label"] for item in label_items) != list(LABELS):
            raise RuntimeError(f"Case {case_id} does not put all labels in first position.")
        expected_original_sequence = list(LABELS)
        for item in label_items:
            actual = [entry["original_option"] for entry in item["sequence_entries"]]
            if actual != expected_original_sequence:
                raise RuntimeError(f"Case {case_id} label control changed content positions.")


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


def rounded_tie_details(probabilities: dict[str, float]) -> dict[str, Any]:
    groups: dict[float, list[str]] = defaultdict(list)
    for label, probability in probabilities.items():
        groups[probability].append(label)
    tied_groups = [labels for labels in groups.values() if len(labels) > 1]
    maximum = max(probabilities.values())
    max_labels = [label for label, value in probabilities.items() if value == maximum]
    return {
        "api_probability_precision": "four decimal places (rounded by Laya)",
        "rounded_probability_tie_groups": tied_groups,
        "rounded_max_probability_tie": len(max_labels) > 1,
        "rounded_max_probability_labels": max_labels,
        "exact_full_precision_tie_determinable": False,
    }


def success_record(
    item: dict[str, Any], request_index: int, seed: int, sampled_ids: list[int], result: Any
) -> dict[str, Any]:
    probabilities = dict(result.details.get("probabilities", {}))
    if tuple(probabilities) != tuple(item["criteria_order"]):
        raise RuntimeError(
            "Laya response probability order differs from the sent criteria order: "
            f"sent={item['criteria_order']} returned={list(probabilities)}"
        )
    predicted_original = item["label_to_original"][result.answer]
    probabilities_by_original = {
        item["label_to_original"][label]: probability
        for label, probability in probabilities.items()
    }
    return {
        "experiment": "laya_label_vs_sequence_position_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_index": request_index,
        "sample_seed": seed,
        "request_shuffle_seed": seed,
        "sampled_case_ids": sampled_ids,
        "original_case_id": item["original_case_id"],
        "sample_rank": item["sample_rank"],
        "control_group": item["control_group"],
        "analysis_groups": item["analysis_groups"],
        "variant": item["variant"],
        "question_text": item["question_text"],
        "criteria": item["criteria"],
        "criteria_order": item["criteria_order"],
        "sequence_entries": item["sequence_entries"],
        "label_to_original": item["label_to_original"],
        "original_to_label": item["original_to_label"],
        "a_sequence_position": item["a_sequence_position"],
        "first_sequence_label": item["first_sequence_label"],
        "original_expected_answer": item["original_expected_answer"],
        "expected_current_label": item["expected_current_label"],
        "decision_backend": "laya",
        "strategy_name": result.strategy_name,
        "status": "success",
        "prediction": result.answer,
        "predicted_original_option": predicted_original,
        "predicted_original_option_text": item["original_choices"][predicted_original],
        "selected_sequence_position": item["criteria_order"].index(result.answer) + 1,
        "is_correct": predicted_original == item["original_expected_answer"],
        "latency_s": result.latency,
        "confidence": result.confidence,
        "answer_confidence": result.details.get("answer_confidence"),
        "probabilities": probabilities,
        "probabilities_by_original_option": probabilities_by_original,
        "tie_check": rounded_tie_details(probabilities),
        "decision_model": result.details.get("decision_model"),
        "routing": result.details.get("routing", {}),
        "usage": result.details.get("usage", {}),
    }


def failure_record(
    item: dict[str, Any], request_index: int, seed: int, sampled_ids: list[int], error: Exception
) -> dict[str, Any]:
    return {
        "experiment": "laya_label_vs_sequence_position_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_index": request_index,
        "sample_seed": seed,
        "request_shuffle_seed": seed,
        "sampled_case_ids": sampled_ids,
        "original_case_id": item["original_case_id"],
        "sample_rank": item["sample_rank"],
        "control_group": item["control_group"],
        "analysis_groups": item["analysis_groups"],
        "variant": item["variant"],
        "question_text": item["question_text"],
        "criteria": item["criteria"],
        "criteria_order": item["criteria_order"],
        "sequence_entries": item["sequence_entries"],
        "label_to_original": item["label_to_original"],
        "original_to_label": item["original_to_label"],
        "a_sequence_position": item["a_sequence_position"],
        "first_sequence_label": item["first_sequence_label"],
        "original_expected_answer": item["original_expected_answer"],
        "expected_current_label": item["expected_current_label"],
        "decision_backend": "laya",
        "strategy_name": "raw-jev",
        "status": "failure",
        "error_type": type(error).__name__,
        "error": str(error),
    }


def records_for_group(records: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    return [record for record in records if group in record["analysis_groups"]]


def consistency_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_case[record["original_case_id"]].append(record)

    fully_consistent = 0
    total_pairs = 0
    agreeing_pairs = 0
    unique_counts: Counter[int] = Counter()
    baseline_flips = 0
    baseline_comparisons = 0
    per_case: list[dict[str, Any]] = []
    for case_id in sorted(by_case):
        case_records = by_case[case_id]
        selected = [record["predicted_original_option"] for record in case_records]
        unique_count = len(set(selected))
        unique_counts[unique_count] += 1
        if unique_count == 1:
            fully_consistent += 1

        for left, right in itertools.combinations(selected, 2):
            total_pairs += 1
            agreeing_pairs += int(left == right)

        baseline = next(record for record in case_records if record["variant"] == "baseline")
        nonbaseline = [record for record in case_records if record["variant"] != "baseline"]
        flips = sum(
            record["predicted_original_option"] != baseline["predicted_original_option"]
            for record in nonbaseline
        )
        baseline_flips += flips
        baseline_comparisons += len(nonbaseline)
        per_case.append(
            {
                "original_case_id": case_id,
                "selected_original_options": {
                    record["variant"]: record["predicted_original_option"]
                    for record in sorted(case_records, key=lambda value: value["variant"])
                },
                "unique_selected_contents": unique_count,
                "fully_consistent": unique_count == 1,
                "changes_from_baseline": flips,
            }
        )

    return {
        "case_count": len(by_case),
        "fully_content_consistent_cases": fully_consistent,
        "fully_content_consistent_rate": fully_consistent / len(by_case),
        "unique_selected_content_count_distribution": dict(sorted(unique_counts.items())),
        "pairwise_content_agreement_rate": agreeing_pairs / total_pairs,
        "changes_from_baseline": baseline_flips,
        "baseline_comparisons": baseline_comparisons,
        "change_from_baseline_rate": baseline_flips / baseline_comparisons,
        "per_case": per_case,
    }


def summarize(
    records: list[dict[str, Any]], sampled_ids: list[int], seed: int, warmups: list[dict[str, Any]]
) -> dict[str, Any]:
    successes = [record for record in records if record["status"] == "success"]
    failures = [record for record in records if record["status"] == "failure"]
    summary: dict[str, Any] = {
        "experiment": "laya_label_vs_sequence_position_v1",
        "sample_seed": seed,
        "request_shuffle_seed": seed,
        "sampled_case_ids_in_sample_order": sampled_ids,
        "unique_request_count": len(records),
        "success_count": len(successes),
        "failure_count": len(failures),
        "warmups": warmups,
        "full_precision_tie_note": (
            "Laya 0.3.20 rounds API probabilities to four decimal places; exact "
            "full-precision score ties cannot be determined from this endpoint."
        ),
    }
    if failures:
        summary["failure_records"] = failures
        return summary

    position_records = records_for_group(successes, "position_control")
    label_records = records_for_group(successes, "label_control")
    if len(position_records) != 80 or len(label_records) != 80:
        raise RuntimeError("Each analysis group must contain 80 successful observations.")

    a_by_position: dict[str, Any] = {}
    for position in range(1, 5):
        subset = [
            record for record in position_records if record["a_sequence_position"] == position
        ]
        selected = sum(record["prediction"] == "A" for record in subset)
        a_by_position[str(position)] = {
            "observations": len(subset),
            "a_selected": selected,
            "a_selected_rate": selected / len(subset),
            "mean_a_probability": sum(record["probabilities"]["A"] for record in subset)
            / len(subset),
        }

    first_by_label: dict[str, Any] = {}
    for label in LABELS:
        subset = [
            record for record in label_records if record["first_sequence_label"] == label
        ]
        selected = sum(record["prediction"] == label for record in subset)
        first_by_label[label] = {
            "observations": len(subset),
            "first_option_selected": selected,
            "first_option_selected_rate": selected / len(subset),
            "mean_first_option_probability": sum(
                record["probabilities"][label] for record in subset
            )
            / len(subset),
        }

    rounded_max_ties = [
        record for record in successes if record["tie_check"]["rounded_max_probability_tie"]
    ]
    any_rounded_ties = [
        record for record in successes if record["tie_check"]["rounded_probability_tie_groups"]
    ]
    summary["position_control"] = {
        "a_selection_by_sequence_position": a_by_position,
        "content_consistency": consistency_summary(position_records),
    }
    summary["label_control"] = {
        "first_option_selection_by_label": first_by_label,
        "content_consistency": consistency_summary(label_records),
    }
    summary["rounded_probability_ties"] = {
        "requests_with_any_rounded_probability_tie": len(any_rounded_ties),
        "requests_with_rounded_max_probability_tie": len(rounded_max_ties),
        "rounded_max_tie_request_indices": [record["request_index"] for record in rounded_max_ties],
    }
    return summary


def main() -> int:
    args = parse_args()
    case_path = Path(args.cases)
    output_path = Path(args.output)
    if not case_path.is_absolute():
        case_path = PROJECT_DIR / case_path
    if not output_path.is_absolute():
        output_path = PROJECT_DIR / output_path
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    if not summary_path.is_absolute():
        summary_path = PROJECT_DIR / summary_path
    for path in (output_path, summary_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    cases = load_and_validate_cases(case_path)
    items, sampled_ids = build_work_items(cases, args.seed)
    print(json.dumps({"sample_seed": args.seed, "sampled_case_ids": sampled_ids}), flush=True)

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with output_path.open("x", encoding="utf-8") as output:
        for request_index, item in enumerate(items, 1):
            try:
                result = strategy.classify(item["question_text"], item["criteria"])
                record = success_record(item, request_index, args.seed, sampled_ids, result)
            except (SystemOneClientError, ValueError) as error:
                record = failure_record(item, request_index, args.seed, sampled_ids, error)
            records.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            if request_index % 20 == 0 or request_index == len(items):
                failure_count = sum(value["status"] == "failure" for value in records)
                print(
                    f"progress={request_index}/{len(items)} "
                    f"success={len(records) - failure_count} failures={failure_count}",
                    flush=True,
                )

    summary = summarize(records, sampled_ids, args.seed, warmups)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"output={output_path}", flush=True)
    print(f"summary={summary_path}", flush=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
