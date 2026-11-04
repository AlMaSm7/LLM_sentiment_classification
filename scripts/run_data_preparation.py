import argparse
from pathlib import Path

from llm_project.data_preparation import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"


def parse_args() -> argparse.Namespace:
    """Read experiment arguments."""

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

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(project_root=PROJECT_ROOT, config_path=CONFIG_PATH, stage=args.stage)


if __name__ == "__main__":
    main()
