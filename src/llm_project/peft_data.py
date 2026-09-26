"""Data loading and preprocessing for PEFT experiments."""

from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

VALID_CONDITIONS = {
    "english",
    "target",
    "bilingual",
}


def condition_suffix(
    language: str,
    condition: str,
) -> str:
    """Return the filename suffix for a PEFT condition."""

    if condition == "english":
        return "en"

    if condition == "target":
        return language

    if condition == "bilingual":
        return "bilingual"

    raise ValueError(f"Unknown condition: {condition}. Choose one of {sorted(VALID_CONDITIONS)}.")


def get_train_dev_files(
    project_root: Path,
    language: str,
    condition: str,
) -> tuple[Path, Path]:
    """Return the train and dev files for one PEFT condition."""

    suffix = condition_suffix(
        language,
        condition,
    )

    data_dir = project_root / "data" / "processed" / language

    train_file = data_dir / f"train_{suffix}.jsonl"

    dev_file = data_dir / f"dev_{suffix}.jsonl"

    check_files_exist(
        train_file,
        dev_file,
    )

    return train_file, dev_file


def get_evaluation_files(
    project_root: Path,
    language: str,
    condition: str,
    split: str,
) -> dict[str, Path]:
    """Return the required evaluation files."""

    if split not in {"dev", "test"}:
        raise ValueError("split must be 'dev' or 'test'.")

    data_dir = project_root / "data" / "processed" / language

    if condition == "target":
        files = {language: (data_dir / f"{split}_{language}.jsonl")}

    elif condition in {
        "english",
        "bilingual",
    }:
        files = {
            "en": (data_dir / f"{split}_en.jsonl"),
            language: (data_dir / f"{split}_{language}.jsonl"),
        }

    else:
        raise ValueError(f"Unknown condition: {condition}")

    check_files_exist(*files.values())

    return files


def check_files_exist(
    *paths: Path,
) -> None:
    """Check that all requested files exist."""

    missing = [path for path in paths if not path.exists()]

    if missing:
        raise FileNotFoundError("\n".join(str(path) for path in missing))


def load_jsonl(
    path: Path,
) -> Dataset:
    """Load one JSONL file as a Hugging Face Dataset."""

    check_files_exist(path)

    return load_dataset(
        "json",
        data_files=str(path),
        split="train",
    )


def create_user_message(
    example: dict[str, Any],
) -> str:
    """Combine instruction and subtitle text."""

    return f"{example['instruction']}\n\nText:\n{example['input']}"


def prepare_training_dataset(
    dataset: Dataset,
    tokenizer: Any,
    max_length: int,
) -> Dataset:
    """Prepare a dataset for supervised language-model training."""

    def tokenize(
        example: dict[str, Any],
    ) -> dict[str, Any]:

        user = {
            "role": "user",
            "content": create_user_message(example),
        }

        assistant = {
            "role": "assistant",
            "content": example["output"],
        }

        # Create the prompt without the answer.
        prompt_text = tokenizer.apply_chat_template(
            [user],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Create the full conversation including
        # the correct answer.
        full_text = tokenizer.apply_chat_template(
            [user, assistant],
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_tokens = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )

        full_tokens = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )

        input_ids = list(full_tokens["input_ids"])

        attention_mask = list(full_tokens["attention_mask"])

        labels = input_ids.copy()

        prompt_length = min(
            len(prompt_tokens["input_ids"]),
            len(labels),
        )

        # Ignore prompt tokens in the loss.
        labels[:prompt_length] = [-100] * prompt_length

        has_answer = any(label != -100 for label in labels)

        return {
            "input_ids": input_ids,
            "attention_mask": (attention_mask),
            "labels": labels,
            "has_answer": has_answer,
        }

    dataset = dataset.map(
        tokenize,
        remove_columns=(dataset.column_names),
    )

    # Remove examples where truncation
    # removed the complete answer.
    dataset = dataset.filter(lambda example: example["has_answer"])

    return dataset.remove_columns(["has_answer"])
