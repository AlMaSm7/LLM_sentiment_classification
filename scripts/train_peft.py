import argparse
from pathlib import Path

import yaml
from peft import PeftModel

from llm_project.peft_training import load_jsonl, resolve_dataset_paths, train
from llm_project.peft_model import build_model_and_tokenizer
from llm_project.peft_evaluation import parse_prediction, labels_to_binary, compute_metrics
from llm_project.peft_finetuning import run_inference


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

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
    
    # 1. Train the model
    train(train_data, dev_data, config)

    # 2. Load the fine-tuned model for evaluation
    print("Loading fine-tuned model for evaluation...")
    base_model, tokenizer = build_model_and_tokenizer(config)
    
    # Access the output directory from the config structure
    output_dir = config.get("output", {}).get("root_dir", "outputs/peft")
    model = PeftModel.from_pretrained(base_model, output_dir)
    model.eval()

    # 3. Run Inference on Test Data
    print("Running inference...")
    raw_predictions = run_inference(model, tokenizer, test_data)

    # 4. Compute Metrics
    parsed_preds = [parse_prediction(p) for p in raw_predictions]
    true_labels = test_data["output"].apply(parse_prediction).tolist()

    y_true = labels_to_binary(true_labels)
    y_pred = labels_to_binary(parsed_preds)

    metrics = compute_metrics(y_true, y_pred)
    
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    print("Starting PEFT training and evaluation script...")
    main()
