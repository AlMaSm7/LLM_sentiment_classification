import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import wandb

from llm_project.peft_training import (
    build_model,
    get_model_dir,
    load_config,
    load_training_data,
    prepare_dataset,
    save_model,
    train_model,
)

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

    # wandb.init(
    #     project=config["wandb_project"],
    #     mode="online" if config.get("use_wandb", True) else "disabled",
    #     name=run_name,
    #     config={
    #         "language": args.language,
    #         "condition": args.condition,
    #         "model": config["model"]["name"],
    #         "rank": config["peft"]["rank"],
    #         "alpha": config["peft"]["alpha"],
    #         "dropout": config["peft"]["dropout"],
    #         "target_modules": config["peft"]["target_modules"],
    #         "learning_rate": config["training"]["learning_rate"],
    #         "batch_size": config["training"]["batch_size"],
    #         "gradient_accumulation_steps": (config["training"]["gradient_accumulation_steps"]),
    #         "epochs": config["training"]["epochs"],
    #         "max_seq_length": config["training"]["max_seq_length"],
    #         "seed": config["experiment"]["seed"],
    #     },
    # )
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
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # SHOULD BE DONE IN PEFT EVALUATION
    # print("7. Evaluating")
    # metrics = evaluate_model(
    #     model=model,
    #     tokenizer=tokenizer,
    #     language=args.language,
    #     condition=args.condition,
    #     config=config,
    #     split=args.split,
    # )

    # # log_metrics_to_wandb(metrics)

    # print()
    # print("Metrics:")
    # print(metrics)

    # print()
    # print("8. Running inference")

    # predictions = run_inference(
    #     model=model,
    #     tokenizer=tokenizer,
    #     texts=examples,
    #     language=args.language,
    #     condition=args.condition,
    #     config=config,
    # )

    # for result in predictions:
    #     print()
    #     print(result["text"])
    #     print("->", result["prediction"])

    # log_predictions_to_wandb(predictions)

    # print()
    # print("Experiment finished.")
    wandb.finish()


# def log_metrics_to_wandb(
#     metrics: dict,
# ) -> None:
#     """Log evaluation metrics to W&B."""

#     for language, values in metrics.items():
#         wandb.log(
#             {
#                 f"{language}/micro_f1": values["micro_f1"],
#                 f"{language}/jaccard": values["jaccard"],
#                 f"{language}/macro_f1": values["macro_f1"],
#                 f"{language}/precision": values["precision"],
#                 f"{language}/recall": values["recall"],
#                 f"{language}/hamming_loss": values["hamming_loss"],
#             }
#         )

#         for emotion, score in values["per_emotion_f1"].items():
#             wandb.log(
#                 {
#                     f"{language}/f1_{emotion}": score,
#                 }
#             )


# def log_predictions_to_wandb(
#     predictions: list[dict],
# ) -> None:
#     """Log inference examples to W&B."""

#     table = wandb.Table(
#         columns=[
#             "text",
#             "prediction",
#         ]
#     )

#     for result in predictions:
#         table.add_data(
#             result["text"],
#             result["prediction"],
#         )

#     wandb.log(
#         {
#             "inference_examples": table,
#         }
#     )


if __name__ == "__main__":
    main()
