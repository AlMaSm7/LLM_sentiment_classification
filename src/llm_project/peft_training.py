from pathlib import Path
from typing import Any

import pandas as pd


def resolve_dataset_paths(
    project_root: Path,
    language: str,
    condition: str,
) -> tuple[Path, Path, Path]:
    """Return the prepared train/dev/test files for one PEFT condition."""
    if condition == "english":
        suffix = "en"
    elif condition == "target":
        suffix = language
    elif condition == "bilingual":
        suffix = "bilingual"
    elif condition == "mixed":
        suffix = "mixed"
    else:
        raise ValueError("condition must be 'english', 'target', 'bilingual', or 'mixed'")

    data_dir = project_root / "data" / "processed" / language

    # "mixed" has no dedicated dev/test file (evaluation always uses the
    # per-language dev_en/dev_<language> files directly). Reuse bilingual's
    # dev/test here only as a placeholder for the Trainer's per-epoch check.
    dev_suffix = "bilingual" if condition == "mixed" else suffix
    test_suffix = "bilingual" if condition == "mixed" else suffix

    return (
        data_dir / f"train_{suffix}.jsonl",
        data_dir / f"dev_{dev_suffix}.jsonl",
        data_dir / f"test_{test_suffix}.jsonl",
    )


def load_jsonl(path: Path) -> pd.DataFrame:
    """Load one prepared JSONL dataset."""
    return pd.read_json(path, lines=True)


def format_example(example: dict[str, Any]) -> str:
    """Example placeholder for model-specific prompt/chat formatting."""
    return f"{example['instruction']}\n\nText: {example['input']}\n\nAnswer: {example['output']}"


def train(
    model: Any,
    tokenizer: Any,
    train_data: pd.DataFrame,
    dev_data: pd.DataFrame,
    config: dict[str, Any],
):
    """Placeholder for the chosen PEFT training implementation."""
    raise NotImplementedError("Implement the PEFT training loop or Trainer.")
