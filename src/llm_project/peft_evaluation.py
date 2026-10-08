import datetime
import json
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)

from llm_project.constants import EMOTIONS
from llm_project.utils import load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_evaluation_paths(
    language: str,
    condition: str,
    split: str,
) -> dict[str, Path]:
    """Find datasets used for evaluation."""

    data_dir = PROJECT_ROOT / "data" / "processed" / language

    # Target-only training is evaluated on the target language
    if condition == "target":
        return {language: data_dir / f"{split}_{language}.jsonl"}

    # English-only and bilingual models are evaluated on English AND the target language
    return {
        "en": data_dir / f"{split}_en.jsonl",
        language: data_dir / f"{split}_{language}.jsonl",
    }


def create_prompt(
    example: dict[str, Any],
) -> str:
    """Create the user prompt."""

    return f"{example['instruction']}\n\nText:\n{example['input']}"


def parse_labels(text: str) -> list[str]:
    """Extract valid emotion labels from model output."""

    match = re.search(r"\[[^\]]*\]", text)

    if match is None:
        return []

    try:
        labels = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    if not isinstance(labels, list):
        return []

    return [label for label in labels if isinstance(label, str) and label in EMOTIONS]


def labels_to_binary(
    labels: list[list[str]],
) -> np.ndarray:
    """Convert emotion names into binary vectors."""

    return np.asarray([[int(emotion in row) for emotion in EMOTIONS] for row in labels])


def compute_metrics(
    true_labels: list[list[str]],
    predictions: list[list[str]],
) -> dict[str, Any]:
    """Calculate required multi-label metrics."""

    y_true = labels_to_binary(true_labels)
    y_pred = labels_to_binary(predictions)

    emotion_f1 = np.asarray(
        f1_score(
            y_true,
            y_pred,
            average=None,
            zero_division=0,
        ),
        dtype=float,
    )

    return {
        "micro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "jaccard": float(
            jaccard_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "hamming_loss": float(
            hamming_loss(
                y_true,
                y_pred,
            )
        ),
        "per_emotion_f1": {
            emotion: float(score)
            for emotion, score in zip(
                EMOTIONS,
                emotion_f1,
                strict=True,
            )
        },
    }


@torch.inference_mode()  # PyTorch doesn't store gradient information with this
def evaluate_dataset(
    model,
    tokenizer,
    dataset: Dataset,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate predictions and calculate metrics."""

    tokenizer.padding_side = "left"
    model.config.use_cache = True
    model.eval()

    rows = cast(
        list[dict[str, Any]],
        list(dataset),
    )

    true_labels = []
    predictions = []

    batch_size = config["generation"]["batch_size"]

    for start in range(
        0,
        len(rows),
        batch_size,
    ):
        batch = rows[start : start + batch_size]

        prompts = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": create_prompt(example),
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for example in batch
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config["training"]["max_seq_length"],
            add_special_tokens=False,
        )

        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        generated = model.generate(
            **inputs,
            max_new_tokens=config["generation"]["max_new_tokens"],
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # remove the prompt tokens and only keep the generated answer
        generated = generated[:, inputs["input_ids"].shape[1] :]

        texts = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        for example, text in zip(
            batch,
            texts,
            strict=True,
        ):
            true_labels.append(parse_labels(example["output"]))

            predictions.append(parse_labels(text))

    return compute_metrics(
        true_labels,
        predictions,
    )


def evaluate_model(
    model,
    tokenizer,
    language: str,
    condition: str,
    config: dict[str, Any],
    split: str = "dev",
) -> dict[str, Any]:
    """Evaluate and save results."""

    paths = get_evaluation_paths(
        language,
        condition,
        split,
    )

    all_metrics = {}

    for eval_language, path in paths.items():
        print(f"Evaluating: {eval_language}")
        print(f"Taking {path}")

        dataset = load_jsonl(path)

        all_metrics[eval_language] = evaluate_dataset(
            model,
            tokenizer,
            dataset,
            config,
        )

    metrics_dir = (
        PROJECT_ROOT / "outputs" / "metrics" / language / condition / config["experiment"]["name"]
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    # generate a timestamp for the metrics file
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = metrics_dir / f"{split}_{time_stamp}_metrics.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_metrics,
            file,
            indent=2,
        )

    print(f"Metrics saved: {output_path}")

    return all_metrics
