"""Run one complete PEFT experiment."""

import argparse

from llm_project.peft_evaluation import (
    evaluate_model,
)
from llm_project.peft_inference import (
    run_inference,
)
from llm_project.peft_training import (
    build_model,
    get_model_dir,
    load_config,
    load_training_data,
    prepare_dataset,
    save_model,
    train_model,
)


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


def get_inference_examples(
    language: str,
) -> list[str]:
    """Small examples for manually checking predictions."""

    if language == "de":
        return [
            "Ich habe große Angst davor, was morgen passieren könnte.",
            "Ich bin unglaublich glücklich, dich wiederzusehen!",
            "Ich kann nicht glauben, dass du mir das angetan hast.",
        ]

    return [
        "Jag är väldigt rädd för vad som kan hända imorgon.",
        "Jag är så glad att se dig igen!",
        "Jag kan inte tro att du gjorde så mot mig.",
    ]


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

    print("5. Training")
    model_dir = get_model_dir(
        args.language,
        args.condition,
        config,
    )

    trainer = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        config=config,
        model_dir=model_dir,
    )

    print("6. Saving model")
    save_model(
        trainer=trainer,
        tokenizer=tokenizer,
        config=config,
        model_dir=model_dir,
        language=args.language,
        condition=args.condition,
    )

    print("7. Evaluating")
    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        language=args.language,
        condition=args.condition,
        config=config,
        split=args.split,
    )

    print()
    print("Metrics:")
    print(metrics)

    print()
    print("8. Running inference")

    examples = get_inference_examples(args.language)

    predictions = run_inference(
        model=model,
        tokenizer=tokenizer,
        texts=examples,
        language=args.language,
        condition=args.condition,
        config=config,
    )

    for result in predictions:
        print()
        print(result["text"])
        print("->", result["prediction"])

    print()
    print("Experiment finished.")


if __name__ == "__main__":
    main()
