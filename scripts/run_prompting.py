import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from llm_project.single_model_prediction.evaluation_metrics import evaluate_results_dataframe
from llm_project.single_model_prediction.prompt_model import load_inference_model, run_inference
from llm_project.utils import load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Read experiment arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--language",
        choices=["de", "sv"],
        required=True,
    )

    parser.add_argument(
        "--target",
        "--condition",
        dest="target",
        choices=["english", "target"],
        required=True,
    )

    parser.add_argument("--split", choices=["dev", "test"], default="dev")

    parser.add_argument(
        "--mode",
        choices=["zero_shot", "five_shot"],
        required=True,
        help="Execute zero-shot or five-shot prompting.",
    )

    return parser.parse_args()


def get_data_sets(
    language: str,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the training, development, and test datasets for inference as DataFrames.
    """
    base_dir = PROJECT_ROOT / "data" / "processed" / language
    target_language = language if target == "target" else "en"

    if target == "bilingual":
        data_suffix = "bilingual"
    else:
        data_suffix = target_language

    train_data = load_jsonl(base_dir / f"train_{data_suffix}.jsonl").to_pandas()
    dev_data = load_jsonl(base_dir / f"dev_{data_suffix}.jsonl").to_pandas()
    test_data = load_jsonl(base_dir / f"test_{data_suffix}.jsonl").to_pandas()

    return train_data, dev_data, test_data


def main():
    """
    Main function to run the inference script.
    """
    # Load configuration
    config_path = PROJECT_ROOT / "configs" / "prompting.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    args = parse_args()
    # update config with command-line arguments
    config["language"] = args.language
    config["target"] = args.target
    artifact_name = config["artifact_setting"]

    if "run" not in config:
        config["run"] = {}
    config["run"]["mode"] = args.mode

    # Load model and tokenizer
    print(f"Loading model: {config['model']['name']}")
    model, tokenizer = load_inference_model(config["model"]["name"])

    train_data, dev_data, test_data = get_data_sets(args.language, args.target)

    if args.split == "dev":
        eval_data = dev_data
    else:
        eval_data = test_data

    print(f"Running inference on {args.split} set with {len(eval_data)} samples...")
    results = run_inference(config, model, tokenizer, train_data, eval_data)
    results_df = pd.DataFrame(results)

    results_dir = PROJECT_ROOT / "outputs" / "predictions" / "prompting" / f"{args.mode}"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"{args.split}_{artifact_name}_predictions.csv"
    results_df.to_csv(csv_path, index=False)

    print(f"Predictions saved to {csv_path}")

    metrics = evaluate_results_dataframe(results_df)

    metrics_path = results_dir / f"{args.split}_{artifact_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {metrics_path}")
    print("\nPrimary Metrics:")
    print(json.dumps(metrics["primary"], indent=2))


if __name__ == "__main__":
    main()
