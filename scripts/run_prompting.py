import argparse
from pathlib import Path

from llm_project.prompting import run_prompting_experiments

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot and five-shot emotion classification"
    )
    parser.add_argument("--config", default="configs/prompting.yaml")
    parser.add_argument(
        "--language",
        choices=["de", "sv"],
        help="target language (overrides the default in prompting.yaml)",
    )
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--select-ids", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["primary", "secondary"],
        help="model roles to evaluate",
    )
    parser.add_argument("--limit", type=int, help="evaluate only the first N examples")
    parser.add_argument("--skip-existing", action="store_true")

    args = parser.parse_args()
    if args.split == "test" and args.sensitivity:
        parser.error("--sensitivity is only allowed on DEV")
    if args.split == "test" and args.limit is not None:
        parser.error("--limit is not allowed on TEST")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    run_prompting_experiments(
        project_root=PROJECT_ROOT,
        config_path=args.config,
        language=args.language,
        split=args.split,
        sensitivity=args.sensitivity,
        select_ids=args.select_ids,
        model_roles=args.models,
        limit=args.limit,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
