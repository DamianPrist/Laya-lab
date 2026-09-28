from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiment import ExperimentRunner, TestCase, load_cases
from models import JevDecision, LLMDecision
from strategies import LLMOnlyStrategy, RawJevStrategy


class FakeJevClient:
    def __init__(self) -> None:
        self.seen_choices: list[dict[str, str]] = []

    def classify(self, state: object, choices: dict[str, str]) -> JevDecision:
        self.seen_choices.append(choices)
        answer = next(iter(choices))
        return JevDecision(
            answer=answer,
            confidence=0.8,
            probabilities={choice_id: 1 / len(choices) for choice_id in choices},
            latency=0.01,
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 2},
        )


class FakeDeepSeekClient:
    def __init__(self) -> None:
        self.seen_choices: list[dict[str, str]] = []

    def classify(self, user_text: str, choices: dict[str, str]) -> LLMDecision:
        self.seen_choices.append(choices)
        return LLMDecision(
            next(iter(choices)),
            0.02,
            "deepseek-test",
            {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )


class StrategyTests(unittest.TestCase):
    def test_experiment_strategies_share_classify_interface(self) -> None:
        choices = {"A": "refund", "B": "keep purchase"}
        strategies = [
            RawJevStrategy(FakeJevClient()),  # type: ignore[arg-type]
            LLMOnlyStrategy(FakeDeepSeekClient()),  # type: ignore[arg-type]
        ]
        results = [strategy.classify("please refund", choices) for strategy in strategies]
        self.assertEqual([result.answer for result in results], ["A"] * 2)

    def test_runner_uses_only_strategy_interface_and_saves_jsonl(self) -> None:
        strategy = RawJevStrategy(FakeJevClient())  # type: ignore[arg-type]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            runner = ExperimentRunner({"any-key": strategy}, output, "laya")
            runner.run("hello", {"YES": "yes", "NO": "no"})
            record = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(record["input"], "hello")
        self.assertEqual(record["decision_backend"], "laya")
        self.assertEqual(record["choices"], {"YES": "yes", "NO": "no"})
        self.assertEqual(record["results"][0]["answer"], "YES")

    def test_batch_passes_two_distinct_choice_sets_to_both_routes(self) -> None:
        decision = FakeJevClient()
        llm = FakeDeepSeekClient()
        strategies = {
            "raw-jev": RawJevStrategy(decision),  # type: ignore[arg-type]
            "llm-only": LLMOnlyStrategy(llm),  # type: ignore[arg-type]
        }
        cases = [
            TestCase(1, "Pick a number", {"A": "one", "B": "two"}, "A"),
            TestCase(
                2,
                "Pick a color",
                {"RED": "red", "GREEN": "green", "BLUE": "blue"},
                "RED",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            runs = ExperimentRunner(strategies, output, "laya").run_batch(cases)

        expected_choice_sets = [case.choices for case in cases]
        self.assertEqual(decision.seen_choices, expected_choice_sets)
        self.assertEqual(llm.seen_choices, expected_choice_sets)
        self.assertEqual(
            [[result.answer for result in results] for _, results in runs],
            [["A", "A"], ["RED", "RED"]],
        )
        self.assertEqual(
            runs[0][1][1].details["usage"],
            {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )

    def test_load_cases_rejects_expected_answer_outside_choices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "id": 1,
                            "text": "question",
                            "choices": {"A": "first", "B": "second"},
                            "expected_answer": "C",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_cases(path)


if __name__ == "__main__":
    unittest.main()

