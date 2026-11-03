import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from llm_project.peft_training import (
    build_model,
    get_model_dir,
    load_config,
    load_training_data,
    prepare_dataset,
    save_model,
    train_model,
)
from llm_project.wandb_logging import wandb_run

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
        "--split",
        choices=["dev", "test"],
        default="dev",
    )

    return parser.parse_args()


def main() -> None:
    """Run training, evaluation and inference."""

    args = parse_args()

    print()
    print("PEFT experiment")
    print(f"Language:  {args.language}")
    print(f"Condition: {args.condition}")
    print()

    print("1. Loading config")
    config = load_config()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = f"{args.language}_{args.condition}_{config['experiment']['name']}_{timestamp}"

    print("2. Loading data")
    train_data, dev_data = load_training_data(
        language=args.language,
        condition=args.condition,
    )

    print("3. Building model")
    model, tokenizer = build_model(config)

    print("4. Preparing data")
    train_dataset = prepare_dataset(
        train_data,
        tokenizer,
        config["training"]["max_seq_length"],
    )

    dev_dataset = prepare_dataset(
        dev_data,
        tokenizer,
        config["training"]["max_seq_length"],
    )

    # initialize wandb only if enabled in the config
    with wandb_run(
        config,
        language=args.language,
        condition=args.condition,
    ) as run:
        if run is not None:
            run.name = run_name

        print("5. Training")
        model_dir = get_model_dir(
            args.language,
            args.condition,
            config,
        )

        # measure the actual fine-tuning time
        training_started = time.perf_counter()
        trainer = train_model(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            config=config,
            model_dir=model_dir,
            experiment=run_name,
        )

        training_seconds = time.perf_counter() - training_started

        print("6. Saving model")
        save_model(
            trainer=trainer,
            tokenizer=tokenizer,
            config=config,
            model_dir=model_dir,
            language=args.language,
            condition=args.condition,
        )

        # save runtime and peak gpu allocation for the cost comparison
        metadata_path = model_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["training_seconds"] = round(training_seconds, 3)
        metadata["peak_gpu_memory_gb"] = (
            round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None
        )
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        # log runtime and gpu memory to wandb
        if run is not None:
            run.log(
                {
                    "training/seconds": metadata["training_seconds"],
                    "training/peak_gpu_memory_gb": metadata["peak_gpu_memory_gb"],
                }
            )


if __name__ == "__main__":
    main()
