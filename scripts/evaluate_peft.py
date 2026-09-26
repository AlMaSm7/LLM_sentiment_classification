"""Evaluate a trained LoRA model."""

import argparse
import json
from pathlib import Path

import yaml

from llm_project.peft_data import (
    get_evaluation_files,
    load_jsonl,
)
from llm_project.peft_evaluation import (
    evaluate,
)
from llm_project.peft_model import (
    load_tokenizer,
    load_trained_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Read evaluation arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    """Evaluate one trained PEFT experiment."""

    args = parse_args()

    run_dir = PROJECT_ROOT / args.run_dir

    # -------------------------
    # Load run information
    # -------------------------

    with (run_dir / "config_used.yaml").open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    with (run_dir / "metadata.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    language = metadata["language"]

    condition = metadata["condition"]

    # -------------------------
    # Load trained model
    # -------------------------

    tokenizer = load_tokenizer(
        config,
        padding_side="left",
    )

    model = load_trained_model(
        config,
        run_dir / "adapter",
    )

    # -------------------------
    # Evaluation files
    # -------------------------

    files = get_evaluation_files(
        PROJECT_ROOT,
        language,
        condition,
        args.split,
    )

    all_metrics = {}

    for (
        evaluation_language,
        path,
    ) in files.items():
        print(f"Evaluating {evaluation_language}")

        metrics, predictions = evaluate(
            model=model,
            tokenizer=tokenizer,
            dataset=load_jsonl(path),
            batch_size=config["generation"]["batch_size"],
            max_length=config["training"]["max_seq_length"],
            max_new_tokens=config["generation"]["max_new_tokens"],
        )

        all_metrics[evaluation_language] = metrics

        predictions_path = run_dir / (f"{args.split}_predictions_{evaluation_language}.json")

        with predictions_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                predictions,
                file,
                indent=2,
                ensure_ascii=False,
            )

    metrics_path = run_dir / f"{args.split}_metrics.json"

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_metrics,
            file,
            indent=2,
        )

    print()
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
