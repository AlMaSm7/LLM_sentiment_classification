import ast
import json
import warnings

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

from llm_project.constants import EMOTIONS


def ensure_list(val) -> list[str]:
    """Safely converts string representations (JSON or Python literals) to a list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                return []
    return []


def check_if_malformed(raw_prediction: str) -> bool:
    """
    Determines if the raw LLM output was malformed.
    """
    try:
        parsed = json.loads(raw_prediction)
        if not isinstance(parsed, list):
            return True
        if not all(isinstance(x, str) for x in parsed):
            return True
        return False
    except (json.JSONDecodeError, TypeError):
        return True


def calculate_multi_label_metrics(
    y_true: list[list[str]], y_pred: list[list[str]], malformed_count: int
) -> dict:
    """
    Computes all required primary, secondary, and diagnostic metrics.
    """
    total_samples = len(y_true)
    mlb = MultiLabelBinarizer(classes=EMOTIONS)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_true_binary = mlb.fit_transform(y_true)
        y_pred_binary = mlb.transform(y_pred)

    micro_f1 = f1_score(y_true_binary, y_pred_binary, average="micro", zero_division=0)
    sample_jaccard = jaccard_score(y_true_binary, y_pred_binary, average="samples", zero_division=0)

    macro_f1 = f1_score(y_true_binary, y_pred_binary, average="macro", zero_division=0)
    micro_precision = precision_score(
        y_true_binary, y_pred_binary, average="micro", zero_division=0
    )
    micro_recall = recall_score(y_true_binary, y_pred_binary, average="micro", zero_division=0)
    hamming = hamming_loss(y_true_binary, y_pred_binary)

    exact_match = accuracy_score(y_true_binary, y_pred_binary)
    malformed_rate = malformed_count / total_samples if total_samples > 0 else 0.0

    report = classification_report(
        y_true_binary, y_pred_binary, target_names=EMOTIONS, output_dict=True, zero_division=0
    )

    return {
        "primary": {"micro_f1": micro_f1, "sample_jaccard": sample_jaccard},
        "secondary": {
            "macro_f1": macro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "hamming_loss": hamming,
        },
        "diagnostic": {
            "exact_match_ratio": exact_match,
            "malformed_output_rate": malformed_rate,
            "per_emotion": {emotion: report[emotion] for emotion in EMOTIONS},
        },
    }


def evaluate_results_dataframe(results_df: pd.DataFrame) -> dict:
    """
    Safely extracts lists from the DataFrame and computes metrics.
    """
    y_true = results_df["true_labels"].apply(ensure_list).tolist()
    y_pred = results_df["normalized_prediction"].apply(ensure_list).tolist()
    
    malformed_count = int(results_df["prediction"].apply(check_if_malformed).sum())
    
    return calculate_multi_label_metrics(y_true, y_pred, malformed_count)