"""Train one LoRA experiment."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from llm_project.peft_data import (
    get_train_dev_files,
    load_jsonl,
    prepare_training_dataset,
)
from llm_project.peft_model import (
    build_lora_model,
    load_tokenizer,
)

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
        "--condition",
        choices=[
            "english",
            "target",
            "bilingual",
        ],
        required=True,
    )

    parser.add_argument(
        "--config",
        default="configs/peft.yaml",
    )

    return parser.parse_args()


def main() -> None:
    """Train one PEFT condition."""

    args = parse_args()

    # -------------------------
    # Config
    # -------------------------

    config_path = PROJECT_ROOT / args.config

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    # -------------------------
    # Data
    # -------------------------

    train_file, dev_file = get_train_dev_files(
        PROJECT_ROOT,
        args.language,
        args.condition,
    )

    print(f"Train: {train_file}")
    print(f"Dev:   {dev_file}")

    # -------------------------
    # Model
    # -------------------------

    tokenizer = load_tokenizer(
        config,
        padding_side="right",
    )

    model = build_lora_model(config)

    # -------------------------
    # Tokenization
    # -------------------------

    train_dataset = prepare_training_dataset(
        load_jsonl(train_file),
        tokenizer,
        config["training"]["max_seq_length"],
    )

    dev_dataset = prepare_training_dataset(
        load_jsonl(dev_file),
        tokenizer,
        config["training"]["max_seq_length"],
    )

    # -------------------------
    # Output
    # -------------------------

    run_dir = (
        PROJECT_ROOT
        / config["output"]["root_dir"]
        / args.language
        / args.condition
        / config["experiment"]["name"]
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------
    # Trainer
    # -------------------------

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        learning_rate=(config["training"]["learning_rate"]),
        per_device_train_batch_size=(config["training"]["batch_size"]),
        per_device_eval_batch_size=(config["training"]["batch_size"]),
        gradient_accumulation_steps=(config["training"]["gradient_accumulation_steps"]),
        num_train_epochs=(config["training"]["epochs"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=("eval_loss"),
        greater_is_better=False,
        bf16=config["training"].get(
            "bf16",
            True,
        ),
        fp16=config["training"].get(
            "fp16",
            False,
        ),
        seed=config["experiment"]["seed"],
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=(
            DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                label_pad_token_id=-100,
            )
        ),
    )

    trainer.train()

    # -------------------------
    # Save
    # -------------------------

    adapter_dir = run_dir / "adapter"

    trainer.save_model(str(adapter_dir))

    tokenizer.save_pretrained(adapter_dir)

    with (run_dir / "config_used.yaml").open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )

    metadata = {
        "language": args.language,
        "condition": args.condition,
        "best_checkpoint": (trainer.state.best_model_checkpoint),
        "checkpoint_rule": ("lowest development loss"),
        "seed": config["experiment"]["seed"],
        "hardware": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
    }

    with (run_dir / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print(f"Finished: {run_dir}")


if __name__ == "__main__":
    main()
