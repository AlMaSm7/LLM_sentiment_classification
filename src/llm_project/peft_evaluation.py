import json
from collections.abc import Iterable
import re
from pathlib import Path
import numpy as np
# from sklearn.metrics import (
#     f1_score,
#     hamming_loss,
#     jaccard_score,
#     precision_score,
#     recall_score,
#     accuracy_score,
#     precision_recall_fscore_support
# )
from typing import Any, cast
import sklearn
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


def load_jsonl(path: Path) -> Dataset:
    """Read a .jsonl file (one JSON object per line) into a Dataset."""
    rows = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:  # skip blank lines
                rows.append(json.loads(line))
    return Dataset.from_list(rows)
 
 
# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
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
 
 
def create_prompt(example: dict[str, Any]) -> str:
    """Create the user prompt."""
 
    return f"{example['instruction']}\n\nText:\n{example['input']}"
 
 
# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def parse_prediction(text: str, lenient: bool = False) -> tuple[list[str], bool]:
    """Parse a model output into (permitted labels, is_malformed).
 
    Strict by default, as the spec requires: the whole output must be a JSON
    list. A well-formed "[]" is a valid empty prediction, NOT malformed.
 
    lenient=True additionally tries to pull the first [...] out of surrounding
    text (e.g. 'Sure! ["joy"]'). Those outputs are still flagged as malformed,
    because the model did not follow the required format; only the labels are
    salvaged. Leave this off for the numbers you report unless you document it.
    """
    stripped = text.strip()
    format_ok = True
 
    try:
        labels = json.loads(stripped)
    except json.JSONDecodeError:
        if not lenient:
            return [], True
        match = re.search(r"\[[^\]]*\]", stripped)
        if match is None:
            return [], True
        try:
            labels = json.loads(match.group())
        except json.JSONDecodeError:
            return [], True
        format_ok = False
 
    if not isinstance(labels, list):
        return [], True
 
    cleaned = [x.strip().lower() for x in labels if isinstance(x, str)]
    permitted = [x for x in dict.fromkeys(cleaned) if x in EMOTIONS]  # dedupe, keep order
 
    if labels and not permitted:  # non-empty list with nothing usable
        return [], True
 
    return permitted, not format_ok
 
 
def labels_to_binary(label_sets: Iterable[Iterable[str]]) -> np.ndarray:
    """Convert label lists to an 8-column binary matrix (fixed EMOTIONS order)."""
    return np.asarray(
        [[int(emotion in set(labels)) for emotion in EMOTIONS] for labels in label_sets],
        dtype=int,
    )
 
 
# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """All metrics are computed from the same binarised 8-label arrays."""
 
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    per_emotion = {
        emotion: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, emotion in enumerate(EMOTIONS)
    }
 
    return {
        # primary
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "sample_jaccard": float(
            jaccard_score(y_true, y_pred, average="samples", zero_division=0)
        ),
        # secondary
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),  # lower is better
        # diagnostic
        "exact_match": float(accuracy_score(y_true, y_pred)),
        "per_emotion": per_emotion,
    }
 
 
# --------------------------------------------------------------------------
# Generation + evaluation
# --------------------------------------------------------------------------
@torch.inference_mode()  # PyTorch doesn't store gradient information with this
def evaluate_dataset(
    model,
    tokenizer,
    dataset: Dataset,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate predictions; return (metrics, per-example records)."""
 
    tokenizer.padding_side = "left"
    model.config.use_cache = True
    model.eval()
 
    rows = cast(list[dict[str, Any]], list(dataset))
 
    true_labels: list[list[str]] = []
    predictions: list[list[str]] = []
    malformed_flags: list[bool] = []
    records: list[dict[str, Any]] = []
 
    batch_size = config["generation"]["batch_size"]
 
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
 
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": create_prompt(example)}],
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
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
 
        for example, text in zip(batch, texts, strict=True):
            gold, gold_malformed = parse_prediction(example["output"])
            if gold_malformed:
                # the reference labels must always be clean; if not, the data is broken
                raise ValueError(f"Malformed gold label: {example['output']!r}")
 
            pred, pred_malformed = parse_prediction(text)
 
            true_labels.append(gold)
            predictions.append(pred)
            malformed_flags.append(pred_malformed)
            records.append(
                {
                    "input": example["input"],
                    "gold": gold,
                    "prediction": pred,
                    "raw_output": text,
                    "malformed": pred_malformed,
                }
            )
 
    metrics = compute_metrics(labels_to_binary(true_labels), labels_to_binary(predictions))
    metrics["malformed_output_rate"] = sum(malformed_flags) / len(malformed_flags)
    metrics["n_examples"] = len(rows)
 
    return metrics, records
 
 
def evaluate_model(
    model,
    tokenizer,
    language: str,
    condition: str,
    config: dict[str, Any],
    split: str = "dev",
) -> dict[str, Any]:
    """Evaluate on every language this condition requires and save results."""
 
    paths = get_evaluation_paths(language, condition, split)
    experiment = config["experiment"]["name"]
 
    metrics_dir = PROJECT_ROOT / "outputs" / "metrics" / language / condition / experiment
    predictions_dir = PROJECT_ROOT / "outputs" / "predictions" / language / condition / experiment
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
 
    all_metrics: dict[str, Any] = {"_meta": {"sklearn_version": sklearn.__version__}}
 
    for eval_language, path in paths.items():
        print(f"Evaluating: {eval_language}")
        print(f"Taking {path}")
 
        dataset = load_jsonl(path)
        metrics, records = evaluate_dataset(model, tokenizer, dataset, config)
        all_metrics[eval_language] = metrics
 
        # raw outputs are needed later for the qualitative error analysis
        predictions_path = predictions_dir / f"{split}_{eval_language}_predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
 
    output_path = metrics_dir / f"{split}_metrics.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2)
 
    print(f"Metrics saved: {output_path}")
    print(f"Predictions saved: {predictions_dir}")
 
    return all_metrics