import json
from typing import Any

import numpy as np
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)

from llm_project.utils.constants import EMOTIONS


def parse_true_labels(
    text: str,
) -> list[str]:
    """Parse true labels from the dataset."""

    labels = json.loads(text)

    if not isinstance(labels, list):
        raise ValueError("True labels must be stored as a JSON list.")

    return [
        label.strip().lower()
        for label in labels
        if (isinstance(label, str) and label.strip().lower() in EMOTIONS)
    ]


def parse_prediction(
    text: str,
) -> tuple[list[str], bool]:
    """
    Parse a generated prediction.

    Returns:
        predicted labels
        malformed flag
    """

    text = text.strip()

    try:
        output = json.loads(text)

    except json.JSONDecodeError:
        return [], True

    if not isinstance(output, list):
        return [], True

    predicted_labels = []

    for item in output:
        if not isinstance(item, str):
            continue

        label = item.strip().lower()

        if label in EMOTIONS and label not in predicted_labels:
            predicted_labels.append(label)

    return predicted_labels, False


def labels_to_binary(
    rows: list[list[str]],
) -> np.ndarray:
    """Convert label lists to binary vectors."""

    return np.asarray(
        [[int(emotion in labels) for emotion in EMOTIONS] for labels in rows],
        dtype=int,
    )


def compute_metrics(
    true_labels: list[list[str]],
    predictions: list[list[str]],
) -> dict[str, Any]:
    """Calculate the required multi-label metrics."""

    y_true = labels_to_binary(true_labels)
    y_pred = labels_to_binary(predictions)

    per_emotion = np.asarray(
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
                per_emotion,
                strict=True,
            )
        },
    }


def evaluate_outputs(
    rows: list[dict[str, Any]],
    raw_outputs: list[str],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Parse predictions and evaluate them."""

    if len(rows) != len(raw_outputs):
        raise ValueError("Number of rows and model outputs must match.")

    true_labels = []
    predictions = []
    records = []

    malformed_count = 0

    for row, raw_output in zip(
        rows,
        raw_outputs,
        strict=True,
    ):
        true_example_labels = parse_true_labels(row["output"])

        predicted_labels, malformed = parse_prediction(raw_output)

        true_labels.append(true_example_labels)

        predictions.append(predicted_labels)

        malformed_count += int(malformed)

        records.append(
            {
                "alignment_id": row.get("alignment_id"),
                "input": row["input"],
                "true_labels": (true_example_labels),
                "prediction": (predicted_labels),
                "raw_prediction": (raw_output),
                "malformed": malformed,
            }
        )

    metrics = compute_metrics(
        true_labels,
        predictions,
    )

    metrics["malformed_output_rate"] = malformed_count / len(rows) if rows else 0.0

    return metrics, records
