"""Run one PEFT experiment."""

import argparse
from pathlib import Path

from llm_project.peft_evaluation import (
    evaluate_model,
)
from llm_project.peft_training import (
    build_model,
    load_saved_model,
    load_training_data,
    prepare_dataset,
    save_model,
    train_model,
)
from llm_project.utils.data_io import (
    load_yaml,
)
from llm_project.wandb_utils import (
    finish_wandb,
    init_wandb,
    log_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""

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
        choices=[
            "dev",
            "test",
        ],
        default="dev",
    )

    parser.add_argument(
        "--evaluate-only",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    """Run training or evaluation."""

    args = parse_args()

    # 1. Configuration
    config = load_yaml(PROJECT_ROOT / "configs" / "peft.yaml")

    run_name = f"{args.language}_{args.condition}_{config['experiment']['name']}"

    print()
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Model:      {config['model']['name']}")
    print(f"PEFT:       {config['peft']['method']}")
    print(f"Language:   {args.language}")
    print(f"Condition:  {args.condition}")
    print(f"Split:      {args.split}")
    print(f"W&B:        {config['wandb']['enabled']}")

    # 2. W&B
    wandb_run = init_wandb(
        config=config,
        run_name=run_name,
    )

    try:
        # 3A. Load an existing model
        if args.evaluate_only:
            print()
            print("Loading saved model")

            model, tokenizer = load_saved_model(
                language=args.language,
                condition=args.condition,
                config=config,
            )

        # 3B. Train a new model
        else:
            if args.split != "dev":
                raise ValueError(
                    "Training must use the "
                    "development split. "
                    "For test evaluation, use "
                    "--evaluate-only."
                )

            print()
            print("Loading training data")

            train_data, dev_data = load_training_data(
                language=args.language,
                condition=args.condition,
            )

            print()
            print("Building model")

            model, tokenizer = build_model(config)

            print()
            print("Preparing datasets")

            train_dataset = prepare_dataset(
                dataset=train_data,
                tokenizer=tokenizer,
                config=config,
            )

            dev_dataset = prepare_dataset(
                dataset=dev_data,
                tokenizer=tokenizer,
                config=config,
            )

            print()
            print("Training")

            trainer = train_model(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                dev_dataset=dev_dataset,
                language=args.language,
                condition=args.condition,
                config=config,
            )

            print()
            print("Saving model")

            save_model(
                trainer=trainer,
                tokenizer=tokenizer,
                language=args.language,
                condition=args.condition,
                config=config,
            )

        # 4. Evaluation
        print()
        print(f"Evaluating {args.split}")

        metrics = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            language=args.language,
            condition=args.condition,
            split=args.split,
            config=config,
        )

        log_metrics(
            run=wandb_run,
            metrics=metrics,
        )

        print()
        print("PEFT experiment finished.")

    finally:
        finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
