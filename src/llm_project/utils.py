from pathlib import Path

from datasets import Dataset, load_dataset


def load_jsonl(path: Path) -> Dataset:
    """Load one JSONL file."""

    if not path.exists():
        raise FileNotFoundError(path)

    return load_dataset(
        "json",
        data_files=str(path),
        split="train",
    )
