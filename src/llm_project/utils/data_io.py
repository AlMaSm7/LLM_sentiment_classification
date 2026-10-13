import json
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    """Load a YAML file."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def get_processed_path(
    language: str,
    split: str,
    variant: str,
) -> Path:
    """Return a processed dataset path."""

    return PROJECT_ROOT / "data" / "processed" / language / f"{split}_{variant}.jsonl"


def load_jsonl_dataset(
    path: Path,
) -> Dataset:
    """Load JSONL as a Hugging Face Dataset."""

    if not path.exists():
        raise FileNotFoundError(path)

    return load_dataset(
        "json",
        data_files=str(path),
        split="train",
    )


def load_jsonl_records(
    path: Path,
) -> list[dict[str, Any]]:
    """Load JSONL as normal Python dictionaries."""

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return [json.loads(line) for line in file if line.strip()]


def save_json(
    data: Any,
    path: Path,
) -> None:
    """Save data as JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )
