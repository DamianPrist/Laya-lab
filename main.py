from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clients import DeepSeekClient, SystemOneClient, SystemOneClientError
from config import Settings
from experiment import ExperimentRunner, load_cases
from strategies import ClassificationStrategy, LLMOnlyStrategy, RawJevStrategy


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Jev and LLM on multiple-choice questions.")
    parser.add_argument(
        "--strategy",
        choices=("raw-jev", "llm-only", "compare"),
        default="compare",
    )
    parser.add_argument("--text", help="Answer this question instead of prompting interactively.")
    parser.add_argument(
        "--choices",
        help='Choices for an interactive question as JSON, for example {"A":"Yes","B":"No"}.',
    )
    parser.add_argument("--batch", nargs="?", const="cases.json", help="Run selected strategies on JSON cases.")
    parser.add_argument("--output", default="results.jsonl", help="Append results to this JSONL file.")
    parser.add_argument("--check", action="store_true", help="Run a startup/configuration check only.")
    return parser.parse_args()


def build_strategies(
    strategy_name: str, settings: Settings
) -> dict[str, ClassificationStrategy]:
    needs_decision = strategy_name in {"raw-jev", "compare"}
    needs_llm = strategy_name in {"llm-only", "compare"}
    decision = create_decision_client(settings) if needs_decision else None
    llm = (
        DeepSeekClient(
            api_key=settings.deepseek_api_key or "",
            model=settings.deepseek_model or "",
            base_url=settings.deepseek_base_url,
            timeout=settings.request_timeout,
        )
        if needs_llm
        else None
    )
    strategies: dict[str, ClassificationStrategy] = {}
    if strategy_name in {"raw-jev", "compare"}:
        assert decision is not None
        strategies["raw-jev"] = RawJevStrategy(decision)
    if strategy_name in {"llm-only", "compare"}:
        assert llm is not None
        strategies["llm-only"] = LLMOnlyStrategy(llm)
    return strategies


def create_decision_client(settings: Settings) -> SystemOneClient:
    settings.validate_decision_base_url()
    if settings.decision_backend == "laya":
        return SystemOneClient(
            api_key=settings.laya_api_key,
            model=None,
            base_url=settings.laya_base_url,
            timeout=settings.request_timeout,
            service_name="Laya",
        )
    return SystemOneClient(
        api_key=settings.typesafe_api_key,
        model=settings.typesafe_model,
        base_url=settings.typesafe_base_url,
        timeout=settings.request_timeout,
        service_name="TypeSafe Jev",
        discover_model=settings.typesafe_model is None,
    )


def print_check(settings: Settings) -> bool:
    print("Startup check passed: imports, CLI, and project files are available.")
    print(f"Decision backend: {settings.decision_backend_display_name}")
    print(f"Base URL: {settings.decision_base_url}")
    if settings.decision_backend == "laya":
        try:
            data = create_decision_client(settings).health()
            if data.get("status") != "ok":
                raise SystemOneClientError("Laya health check returned a non-OK status.")
        except (SystemOneClientError, ValueError) as exc:
            print("Status: ERROR")
            print(str(exc))
            print("CUDA fallback: $env:LAYA_DEVICE=\"cpu\"; laya-serve")
            return False
        print("Status: OK")
        print(f"Device: {data.get('device', 'unknown')}")
        loaded = data.get("loaded")
        if loaded is not None:
            print(f"Loaded models: {loaded}")
        return True
    print(f"DEEPSEEK_API_KEY: {'set' if settings.deepseek_api_key else 'missing'}")
    print(f"DEEPSEEK_MODEL: {'set' if settings.deepseek_model else 'missing'}")
    print(f"DEEPSEEK_BASE_URL: {settings.deepseek_base_url}")
    print(f"TYPESAFE_API_KEY: {'set' if settings.typesafe_api_key else 'missing'}")
    model_status = "set" if settings.typesafe_model else "auto-discover via GET /v1/models"
    print(f"TYPESAFE_MODEL: {model_status}")
    print("No API request was made.")
    return True


def main() -> int:
    args = parse_args()
    try:
        settings = Settings.load()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if args.check:
        return 0 if print_check(settings) else 1

    strategy_name = args.strategy
    missing = settings.missing_for(strategy_name)
    if missing:
        print(f"Missing configuration: {', '.join(missing)}", file=sys.stderr)
        print("Copy .env.example to .env and fill the required values.", file=sys.stderr)
        return 2

    try:
        strategies = build_strategies(strategy_name, settings)
        print(f"Decision backend: {settings.decision_backend_display_name}")
        print(f"Base URL: {settings.decision_base_url}")
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_DIR / output_path
        runner = ExperimentRunner(strategies, output_path, settings.decision_backend)
        if args.batch:
            case_path = Path(args.batch)
            if not case_path.is_absolute():
                case_path = PROJECT_DIR / case_path
            runner.run_batch(load_cases(case_path))
        else:
            if not args.choices:
                raise ValueError("--choices is required when running without --batch.")
            choices = _parse_choices(args.choices)
            user_text = args.text or input("Question: ").strip()
            if not user_text:
                print("Input cannot be empty.", file=sys.stderr)
                return 2
            runner.run(user_text, choices)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def _parse_choices(raw: str) -> dict[str, str]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("--choices must be a valid JSON object.") from exc
    if not isinstance(data, dict) or not data:
        raise ValueError("--choices must be a non-empty JSON object.")
    if not all(
        isinstance(choice_id, str)
        and choice_id
        and isinstance(description, str)
        and description
        for choice_id, description in data.items()
    ):
        raise ValueError("Choice IDs and descriptions must be non-empty strings.")
    return dict(data)


if __name__ == "__main__":
    raise SystemExit(main())
