from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    jaccard_score,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import MultiLabelBinarizer

from llm_project.constants import EMOTIONS

_LABEL_SET = frozenset(EMOTIONS)


@dataclass(frozen=True)
class ParsedOutput:
    labels: tuple[str, ...]
    malformed: bool
    strict_json: bool

    @property
    def empty(self) -> bool:
        return not self.labels


def normalize(item: str) -> str:
    # normalize case and whitespace without changing label meaning
    return " ".join(item.split()).lower()


def parse_prediction(raw: str) -> ParsedOutput:
    # require a complete json array with permitted english label names
    try:
        values = json.loads(raw.strip())
    except (TypeError, json.JSONDecodeError):
        return ParsedOutput((), malformed=True, strict_json=False)

    if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
        return ParsedOutput((), malformed=True, strict_json=False)

    normalized = {normalize(value) for value in values}
    if normalized - _LABEL_SET:
        return ParsedOutput((), malformed=True, strict_json=False)

    labels = tuple(label for label in EMOTIONS if label in normalized)
    return ParsedOutput(labels, malformed=False, strict_json=True)


def parse_labels(raw: str) -> list[str]:
    return list(parse_prediction(raw).labels)


def gold_labels(output: Any) -> list[str]:
    if isinstance(output, list):
        items = output
    else:
        parsed = parse_prediction(str(output))
        if parsed.malformed:
            raise ValueError(f"Could not parse gold labels: {output!r}")
        return list(parsed.labels)
    normalised = {normalize(x) for x in items}
    unknown = normalised - _LABEL_SET
    if unknown:
        raise ValueError(f"Gold labels outside the label set: {unknown}")
    return [label for label in EMOTIONS if label in normalised]


def binarize(gold, pred):
    # use the same fixed label order for every experiment
    encoder = MultiLabelBinarizer(classes=EMOTIONS)
    return (
        np.asarray(encoder.fit_transform(gold), dtype=np.int8),
        np.asarray(encoder.transform(pred), dtype=np.int8),
    )


def compute_metrics(
    gold: Sequence[Sequence[str]],
    pred: Sequence[Sequence[str]],
    parsed: Sequence[ParsedOutput] | None = None,
) -> dict:
    if len(gold) != len(pred) or not gold:
        raise ValueError("Gold and predictions must have the same nonzero length")

    true, predicted = binarize(gold, pred)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        true, predicted, average="micro", zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        true, predicted, average="macro", zero_division=0
    )
    per_p, per_r, per_f, support = precision_recall_fscore_support(
        true, predicted, average=None, zero_division=0
    )

    per_p = np.asarray(per_p, dtype=float)
    per_r = np.asarray(per_r, dtype=float)
    per_f = np.asarray(per_f, dtype=float)
    support = np.asarray(support, dtype=int)

    metrics = {
        "n": len(gold),
        "micro_f1": float(micro_f1),
        "sample_jaccard": float(jaccard_score(true, predicted, average="samples", zero_division=0)),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "hamming_loss": float(hamming_loss(true, predicted)),
        "exact_match": float(accuracy_score(true, predicted)),
        "gold_label_cardinality": float(np.mean(true.sum(axis=1))),
        "pred_label_cardinality": float(np.mean(predicted.sum(axis=1))),
        "empty_prediction_rate": float(np.mean(predicted.sum(axis=1) == 0)),
        "per_label": {
            label: {
                "precision": float(per_p[i]),
                "recall": float(per_r[i]),
                "f1": float(per_f[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(EMOTIONS)
        },
    }

    # retain the old key for notebooks that already use it
    metrics["jaccard"] = metrics["sample_jaccard"]

    if parsed is not None:
        if len(parsed) != len(gold):
            raise ValueError("Parsed predictions have a different length")
        metrics["malformed_rate"] = float(np.mean([p.malformed for p in parsed]))
        metrics["strict_json_rate"] = float(np.mean([p.strict_json for p in parsed]))
        metrics["wellformed_but_no_label_rate"] = float(
            np.mean([not p.malformed and not p.labels for p in parsed])
        )
    return metrics


def confusion_matrices(gold, pred) -> dict[str, dict[str, int]]:
    true, predicted = binarize(gold, pred)
    matrices = multilabel_confusion_matrix(true, predicted)
    return {
        label: dict(zip(("tn", "fp", "fn", "tp"), map(int, matrices[i].ravel()), strict=True))
        for i, label in enumerate(EMOTIONS)
    }
