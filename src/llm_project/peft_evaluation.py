"""Inference and evaluation for PEFT emotion classification."""

import json
import re
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

from llm_project.peft_data import (
    create_user_message,
)

EMOTIONS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]


def parse_labels(
    text: str,
) -> list[str]:
    """Extract allowed emotion labels from generated text."""

    match = re.search(
        r"\[[^\]]*\]",
        text,
    )

    if match is None:
        return []

    try:
        labels = json.loads(match.group())

    except json.JSONDecodeError:
        return []

    if not isinstance(
        labels,
        list,
    ):
        return []

    return [label for label in labels if (isinstance(label, str) and label in EMOTIONS)]


def labels_to_binary(
    label_sets: list[list[str]],
) -> np.ndarray:
    """Convert emotion names to binary vectors."""

    return np.asarray(
        [[int(emotion in labels) for emotion in EMOTIONS] for labels in label_sets],
        dtype=int,
    )


def compute_metrics(
    gold: list[list[str]],
    predicted: list[list[str]],
) -> dict[str, Any]:
    """Calculate the required multi-label metrics."""

    y_true = labels_to_binary(gold)

    y_pred = labels_to_binary(predicted)

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


@torch.inference_mode()
def evaluate(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Generate predictions and calculate metrics."""

    tokenizer.padding_side = "left"

    rows = cast(
        list[dict[str, Any]],
        list(dataset),
    )

    predictions = []

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
                        "content": (create_user_message(example)),
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
            max_length=max_length,
            add_special_tokens=False,
        )

        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        generated = model.generate(
            **inputs,
            max_new_tokens=(max_new_tokens),
            do_sample=False,
            pad_token_id=(tokenizer.pad_token_id),
            eos_token_id=(tokenizer.eos_token_id),
        )

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
            predictions.append(
                {
                    "input": (example["input"]),
                    "gold": (parse_labels(example["output"])),
                    "prediction": (parse_labels(text)),
                    "raw_prediction": (text),
                }
            )

    metrics = compute_metrics(
        gold=[row["gold"] for row in predictions],
        predicted=[row["prediction"] for row in predictions],
    )

    return metrics, predictions
