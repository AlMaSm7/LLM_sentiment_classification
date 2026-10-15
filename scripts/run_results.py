import argparse
from pathlib import Path

from llm_project.results import (
    LANGUAGES,
    create_report,
    load_metrics,
    required_results,
    select_prompts,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument(
        "--select",
        action="store_true",
        help="freeze prompting choice using DEV metrics",
    )
    parser.add_argument("--lang", choices=["de", "sv", "all"], default="all")
    args = parser.parse_args()
    langs = [lang for lang in LANGUAGES if (Path("data/processed") / lang / "train.jsonl").exists()]
    if args.lang != "all":
        langs = [args.lang]
    if not langs:
        raise FileNotFoundError("No prepared languages found")
    rows = load_metrics(args.split)
    if args.select:
        if args.split != "dev":
            raise ValueError("Prompt selection can only use DEV")
        select_prompts(rows, langs)
    else:
        if args.split == "test":
            rows = required_results(rows, langs)
        else:
            rows = [r for r in rows if r["lang"] in langs]
        create_report(rows, args.split)
