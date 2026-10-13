import argparse
from pathlib import Path

from llm_project.prompting import (
    evaluate_prompting,
    load_model,
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
        "--model",
        choices=[
            "primary",
            "secondary",
        ],
        required=True,
    )

    parser.add_argument(
        "--method",
        choices=[
            "zero_shot",
            "five_shot",
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
        "--prompt-version",
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    """Run one prompting experiment."""

    args = parse_args()

    config = load_yaml(PROJECT_ROOT / "configs" / "prompting.yaml")

    prompt_version = args.prompt_version or config["prompt"]["selected"]

    # once we reach TEST, only the frozen development-selected prompt may be used
    if args.split == "test" and prompt_version != config["prompt"]["selected"]:
        raise ValueError("Test evaluation must use the selected frozen prompt.")

    model_name = config["models"][args.model]["name"]

    run_name = f"prompting_{args.language}_{args.model}_{args.method}_{prompt_version}_{args.split}"

    print()
    print(f"Model:          {model_name}")
    print(f"Language:       {args.language}")
    print(f"Method:         {args.method}")
    print(f"Prompt version: {prompt_version}")
    print(f"Split:          {args.split}")

    wandb_run = init_wandb(
        config=config,
        run_name=run_name,
    )

    try:
        print()
        print("Loading model")

        model, tokenizer = load_model(
            model_key=args.model,
            config=config,
        )

        print()
        print("Running evaluation")

        metrics = evaluate_prompting(
            model=model,
            tokenizer=tokenizer,
            model_key=args.model,
            language=args.language,
            method=args.method,
            split=args.split,
            prompt_version=(prompt_version),
            config=config,
        )

        log_metrics(
            run=wandb_run,
            metrics=metrics,
        )

        print()
        print("Prompting experiment finished.")

    finally:
        finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
