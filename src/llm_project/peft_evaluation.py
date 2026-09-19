import json
from collections.abc import Iterable

import numpy as np
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
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


def parse_prediction(text: str) -> list[str]:
    """Parse a JSON-list model prediction."""
    try:
        labels = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(labels, list):
        return []

    return [label for label in labels if isinstance(label, str) and label in EMOTIONS]


def labels_to_binary(label_sets: Iterable[Iterable[str]]) -> np.ndarray:
    """Convert label lists to an 8-column binary matrix."""
    return np.asarray(
        [[int(emotion in set(labels)) for emotion in EMOTIONS] for labels in label_sets],
        dtype=int,
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, object]:
    """Example metric set used across PEFT experiments."""
    per_emotion = np.asarray(
        f1_score(
            y_true,
            y_pred,
            average=None,
            zero_division=0,
        ),
        dtype=float,
    ).reshape(-1)

    return {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "jaccard_micro": jaccard_score(y_true, y_pred, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "per_emotion_f1": dict(zip(EMOTIONS, map(float, per_emotion.tolist()), strict=True)),
    }
