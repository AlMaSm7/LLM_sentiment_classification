import argparse
from pathlib import Path

import yaml

from llm_project.peft_training import (
    load_jsonl,
    resolve_dataset_paths,
)

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

    print(f"Language: {args.language}")
    print(f"Condition: {args.condition}")
    print(f"Train: {train_file} ({len(train_data):,} examples)")
    print(f"Dev:   {dev_file} ({len(dev_data):,} examples)")
    print(f"Test:  {test_file}")
    print(f"Model: {config['model']['name']}")

    # Example:
    # model, tokenizer = build_model_and_tokenizer(config)
    # train(model, tokenizer, train_data, dev_data, config)
    # then save the model in outputs/models/


if __name__ == "__main__":
    main()
