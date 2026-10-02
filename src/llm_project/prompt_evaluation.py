"""Multi-label metrics for Phase 2.

Align the headline metrics with section 7 of the assignment. If section 7 uses a
metric that is not here, add it to `compute_metrics` so every condition is
scored identically.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from llm_project.prompting import LABELS, ParsedOutput
from sklearn.metrics import (
    f1_score,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import MultiLabelBinarizer


def _binarise(gold, pred):
    mlb = MultiLabelBinarizer(classes=LABELS)
    return mlb.fit_transform(gold), mlb.transform(pred)


def compute_metrics(
    gold: Sequence[Sequence[str]],
    pred: Sequence[Sequence[str]],
    parsed: Sequence[ParsedOutput] | None = None,
) -> dict:
    Y, P = _binarise(gold, pred)
    g_sets = [set(g) for g in gold]
    p_sets = [set(p) for p in pred]

    # Per-example scores. An empty gold set predicted as empty counts as correct.
    sample_f1, jaccard = [], []
    for g, p in zip(g_sets, p_sets):
        if not g and not p:
            sample_f1.append(1.0)
            jaccard.append(1.0)
            continue
        inter = len(g & p)
        sample_f1.append(2 * inter / (len(g) + len(p)))
        jaccard.append(inter / len(g | p))

    prec, rec, f1, support = precision_recall_fscore_support(
        Y, P, labels=range(len(LABELS)), zero_division=0
    )
    metrics = {
        "n": len(gold),
        "micro_f1": float(f1_score(Y, P, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(Y, P, average="macro", zero_division=0)),
        "samples_f1": float(np.mean(sample_f1)),
        "jaccard": float(np.mean(jaccard)),
        "exact_match": float(np.mean([g == p for g, p in zip(g_sets, p_sets)])),
        "gold_label_cardinality": float(np.mean([len(g) for g in g_sets])),
        "pred_label_cardinality": float(np.mean([len(p) for p in p_sets])),
        "empty_prediction_rate": float(np.mean([not p for p in p_sets])),
        "per_label": {
            label: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
    }
    if parsed is not None:
        metrics["malformed_rate"] = float(np.mean([p.malformed for p in parsed]))
        metrics["strict_json_rate"] = float(np.mean([p.strict_json for p in parsed]))
        metrics["wellformed_but_no_label_rate"] = float(
            np.mean([(not p.malformed) and p.empty for p in parsed])
        )
    return metrics


def confusion_matrices(gold, pred) -> dict[str, dict[str, int]]:
    """One 2x2 matrix per emotion (tn, fp, fn, tp)."""
    Y, P = _binarise(gold, pred)
    mats = multilabel_confusion_matrix(Y, P)
    return {
        label: dict(zip(["tn", "fp", "fn", "tp"], map(int, mats[i].ravel())))
        for i, label in enumerate(LABELS)
    }
    
