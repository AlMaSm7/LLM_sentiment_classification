import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml
from peft import PeftModel

import wandb
from llm_project.peft_evaluation import compute_metrics, labels_to_binary, parse_prediction
from llm_project.peft_finetuning import run_inference
from llm_project.peft_model import build_model_and_tokenizer
from llm_project.peft_training import load_jsonl, resolve_dataset_paths, train

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument(
        "--condition",
        choices=["english", "target", "bilingual"],
        required=True,
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def _run_name(config: dict[str, Any], args: argparse.Namespace) -> str:
    configured_name = args.run_name or os.environ.get("WANDB_NAME")
    if configured_name:
        return configured_name.replace("/", "-").replace(" ", "_")

    method = config["peft"]["method"].lower()
    return f"{method}_{args.language}_{args.condition}"


def _output_directories(
    config: dict[str, Any],
    run_name: str,
) -> tuple[Path, Path, Path]:
    output_root = PROJECT_ROOT / config.get("output", {}).get("root_dir", "outputs")
    model_dir = output_root / "models" / run_name
    metrics_dir = output_root / "metrics" / run_name
    predictions_dir = output_root / "predictions" / run_name

    for directory in (model_dir, metrics_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return model_dir, metrics_dir, predictions_dir

# log the outputs to directories and sanity check
def _save_evaluation_outputs(
    test_data,
    raw_predictions: list[str],
    metrics: dict[str, object],
    metrics_dir: Path,
    predictions_dir: Path,
) -> Any:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    parsed_predictions = [parse_prediction(prediction) for prediction in raw_predictions]
    predictions = test_data.copy()
    predictions["raw_prediction"] = raw_predictions
    predictions["predicted_labels"] = parsed_predictions

    predictions.to_json(
        predictions_dir / "predictions.jsonl",
        orient="records",
        lines=True,
        force_ascii=True,
    )
    predictions.to_csv(predictions_dir / "predictions.csv", index=False)

    with (metrics_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return predictions


def _wandb_metrics(metrics: dict[str, object]) -> dict[str, float]:
    flat_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if key != "per_emotion_f1"
    }
    flat_metrics.update(
        {
            f"per_emotion_f1/{emotion}": float(value)
            for emotion, value in metrics["per_emotion_f1"].items()
        }
    )
    return flat_metrics


def main() -> None:
    args = parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    run_name = _run_name(config, args)
    model_dir, metrics_dir, predictions_dir = _output_directories(config, run_name)
    config["output"] = {**config.get("output", {}), "root_dir": str(model_dir)}
    config["wandb_run_name"] = run_name

    train_file, dev_file, test_file = resolve_dataset_paths(
        PROJECT_ROOT,
        args.language,
        args.condition,
    )

    
    train_data = load_jsonl(train_file)
    dev_data = load_jsonl(dev_file)
    test_data = load_jsonl(test_file)

    print(f"Language: {args.language}")
    print(f"Condition: {args.condition}")
    print(f"Train: {train_file} ({len(train_data):,} examples)")
    print(f"Dev:   {dev_file} ({len(dev_data):,} examples)")
    print(f"Test:  {test_file} ({len(test_data):,} examples)")
    print(f"Model: {config['model']['id']}")
    
    wandb_config = dict(config)
    with wandb.init(
        project=config.get("wandb_project", "peft_emotion_classification"),
        name=run_name,
        config=wandb_config,
        reinit="finish_previous",
    ) as run:
        run.config.update(
            {
                "language": args.language,
                "condition": args.condition,
                "model_dir": str(model_dir),
                "metrics_dir": str(metrics_dir),
                "predictions_dir": str(predictions_dir),
            }
        )

        print(f"Run: {run_name}")
        print(f"Models: {model_dir}")
        print(f"Metrics: {metrics_dir}")
        print(f"Predictions: {predictions_dir}")

        train(train_data, dev_data, config)

        print("Loading fine-tuned model for evaluation...")
        base_model, tokenizer = build_model_and_tokenizer(config)
        model = PeftModel.from_pretrained(base_model, model_dir)
        model.eval()

        print("Running inference...")
        raw_predictions = run_inference(model, tokenizer, test_data)
        parsed_preds = [parse_prediction(prediction) for prediction in raw_predictions]
        true_labels = test_data["output"].apply(parse_prediction).tolist()
        metrics = compute_metrics(
            labels_to_binary(true_labels),
            labels_to_binary(parsed_preds),
        )

        predictions = _save_evaluation_outputs(
            test_data,
            raw_predictions,
            metrics,
            metrics_dir,
            predictions_dir,
        )
        run.log(_wandb_metrics(metrics))
        run.log({"predictions": wandb.Table(dataframe=predictions)})
        run.save(str(metrics_dir / "metrics.json"))
        run.save(str(predictions_dir / "predictions.jsonl"))

        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")


if __name__ == "__main__":
    print("Starting PEFT training and evaluation script...")
    main()
