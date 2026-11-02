import argparse
from pathlib import Path

from llm_project.results import generate_combined_report, generate_results

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create tables and plots for the report")
    parser.add_argument("--target", choices=["de", "sv", "both"], default="sv")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--select", action="store_true", help="select prompt variants using DEV")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="explore partial runs without requiring the full experiment matrix",
    )
    args = parser.parse_args()
    if args.select and args.split != "dev":
        parser.error("prompt selection is only allowed on DEV")
    if args.select and args.allow_incomplete:
        parser.error("prompt selection requires complete DEV results")
    return args


def main() -> None:
    args = parse_args()
    if args.target == "both":
        for language in ("de", "sv"):
            has_metrics = any(
                any(
                    (PROJECT_ROOT / "outputs" / "metrics" / family / args.split).glob(
                        f"*__{language}__*.json"
                    )
                )
                for family in ("prompting", "peft")
            )
            if args.allow_incomplete and not has_metrics:
                print(
                    f"No {language.upper()} {args.split.upper()} metrics yet; skipping its individual exports"
                )
                continue
            generate_results(
                PROJECT_ROOT,
                target=language,
                split=args.split,
                select=args.select,
                allow_incomplete=args.allow_incomplete,
            )
        generate_combined_report(PROJECT_ROOT, args.split, args.allow_incomplete)
        return

    generate_results(
        project_root=PROJECT_ROOT,
        target=args.target,
        split=args.split,
        select=args.select,
        allow_incomplete=args.allow_incomplete,
    )


if __name__ == "__main__":
    main()
