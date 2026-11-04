"""Choose which stage of XED/OPUS data preparation to run."""

import argparse
from pathlib import Path

from llm_project.data_preparation import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare English–Swedish emotion data")
    parser.add_argument(
        "--stage",
        choices=("download", "split", "rebalance", "all"),
        required=True,
        help=(
            "download: cache XED/OPUS source files; "
            "split: create processed datasets from cached sources; "
            "rebalance: update only train_bilingual.jsonl; "
            "all: download and split"
        ),
    )
    args = parser.parse_args()
    run_pipeline(project_root=PROJECT_ROOT, config_path=CONFIG_PATH, stage=args.stage)


if __name__ == "__main__":
    main()
