from pathlib import Path
from typing import Any

from llm_project.evaluation import (
    evaluate_outputs,
)
from llm_project.utils.data_io import get_processed_path, load_jsonl_records, save_json
from llm_project.utils.model_helpers import generate_texts
from llm_project.utils.prompts import build_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_evaluation_paths(
    language: str,
    condition: str,
    split: str,
) -> dict[str, Path]:
    """Return datasets required for evaluation."""

    if condition == "target":
        return {
            language: get_processed_path(
                language=language,
                split=split,
                variant=language,
            )
        }

    if condition in {
        "english",
        "bilingual",
    }:
        return {
            "en": get_processed_path(
                language=language,
                split=split,
                variant="en",
            ),
            language: get_processed_path(
                language=language,
                split=split,
                variant=language,
            ),
        }

    raise ValueError(f"Unknown condition: {condition}")


def evaluate_model(
    model,
    tokenizer,
    language: str,
    condition: str,
    split: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a model and save metrics and predictions."""

    paths = get_evaluation_paths(
        language=language,
        condition=condition,
        split=split,
    )

    experiment_name = config["experiment"]["name"]

    metrics_dir = PROJECT_ROOT / "outputs" / "metrics" / language / condition / experiment_name
    predictions_dir = (
        PROJECT_ROOT / "outputs" / "predictions" / language / condition / experiment_name
    )

    all_metrics = {}

    for eval_language, path in paths.items():
        print(f"Evaluating: {eval_language}")

        rows = load_jsonl_records(path)

        prompts = [
            build_prompt(
                text=row["input"],
                instruction=row["instruction"],
            )
            if row["instruction"]
            else build_prompt(text=row["input"])
            for row in rows
        ]

        raw_outputs = generate_texts(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=(config["generation"]["batch_size"]),
            max_length=(config["training"]["max_seq_length"]),
            max_new_tokens=(config["generation"]["max_new_tokens"]),
            do_sample=(config["generation"]["do_sample"]),
        )

        metrics, predictions = evaluate_outputs(
            rows=rows,
            raw_outputs=raw_outputs,
        )

        all_metrics[eval_language] = metrics

        save_json(
            predictions,
            predictions_dir / (f"{split}_{eval_language}.json"),
        )

    metrics_path = metrics_dir / f"{split}_metrics.json"

    save_json(
        all_metrics,
        metrics_path,
    )

    print(f"Metrics saved: {metrics_path}")

    return all_metrics
