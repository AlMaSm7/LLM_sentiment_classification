import json
import re
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import pandas as pd

from llm_project.constants import EMOTIONS, INSTRUCTION_QWEN

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIVE_SHOT_IDS_PATH = PROJECT_ROOT / "configs" / "five_shot_ids.json"


def get_five_shot_ids(config: dict[str, Any], train_dataset: pd.DataFrame) -> list[str]:
    """Load or generate the five-shot IDs for the active language and target."""
    language = config["language"]
    target = config["target"]
    if not isinstance(language, str) or not isinstance(target, str):
        raise ValueError("Five-shot inference requires 'language' and 'target' in the config")

    config_key = f"{language}_{target}"
    if FIVE_SHOT_IDS_PATH.exists():
        with FIVE_SHOT_IDS_PATH.open("r", encoding="utf-8") as file:
            saved_ids = json.load(file)
    else:
        saved_ids = {}

    fixed_ids = saved_ids.get(config_key)
    if not isinstance(fixed_ids, list) or not all(isinstance(item, str) for item in fixed_ids):
        print(f"No saved five-shot IDs found for {config_key}. Generating optimal set...")
        fixed_ids = fetch_best_five_shot_ids(train_dataset)
        saved_ids[config_key] = fixed_ids
        with FIVE_SHOT_IDS_PATH.open("w", encoding="utf-8") as file:
            json.dump(saved_ids, file, indent=2, ensure_ascii=False)
            file.write("\n")
        print(f"Saved five-shot IDs to {FIVE_SHOT_IDS_PATH}")

    return fixed_ids


def fetch_best_five_shot_ids(train_df: pd.DataFrame) -> list[str]:
    """Greedy algorithm to select optimal five-shot examples."""
    selected_indices = []
    covered_emotions = set()

    for _ in range(5):
        best_idx = None
        best_new_coverage = -1

        for idx, row in train_df.iterrows():
            if idx in selected_indices:
                continue

            row_labels = row["output"]
            if isinstance(row_labels, str):
                try:
                    row_labels = json.loads(row_labels)
                except json.JSONDecodeError:
                    row_labels = []

            row_emotions = set(row_labels)
            new_emotions = row_emotions - covered_emotions

            if len(new_emotions) > best_new_coverage:
                best_new_coverage = len(new_emotions)
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            best_labels = train_df.at[best_idx, "output"]
            if isinstance(best_labels, str):
                try:
                    best_labels = json.loads(best_labels)
                except json.JSONDecodeError:
                    best_labels = []
            covered_emotions.update(best_labels)

    return [train_df.at[idx, "alignment_id"] for idx in selected_indices]


def fetch_fixed_five_shot_examples(
    train_df: pd.DataFrame, fixed_ids: list[str]
) -> list[dict[str, str]]:
    """Fetches the training examples corresponding to the provided alignment IDs."""
    fixed_df = train_df[train_df["alignment_id"].isin(fixed_ids)].copy()
    fixed_df["id_category"] = pd.Categorical(
        fixed_df["alignment_id"], categories=fixed_ids, ordered=True
    )
    fixed_df = fixed_df.sort_values("id_category")

    demonstrations = []
    for _, row in fixed_df.iterrows():
        labels = json.loads(row["output"]) if isinstance(row["output"], str) else row["output"]
        label_string = json.dumps(labels, ensure_ascii=False)
        demonstrations.append({"input": row["input"], "output": label_string})

    return demonstrations


def check_output_formatting(output: str) -> list[str]:
    """Parses the JSON output using regex, normalizes it, and filters out unpermitted labels."""
    match = re.search(r"\[.*?\]", output, re.DOTALL)
    if not match:
        return []

    json_str = match.group(0)

    try:
        parsed_output = json.loads(json_str)

        if not isinstance(parsed_output, list) or not all(
            isinstance(e, str) for e in parsed_output
        ):
            return []

        valid_emotions = []
        seen = set()

        for emotion in parsed_output:
            normalized = emotion.strip().lower()
            if normalized in EMOTIONS and normalized not in seen:
                valid_emotions.append(normalized)
                seen.add(normalized)

        return valid_emotions

    except (json.JSONDecodeError, TypeError):
        return []


def prepare_prompt_five_shot(
    json_data: dict[str, Any], five_shot_examples: list[dict[str, str]]
) -> str:
    """Formats the model prompt with the fixed five-shot demonstrations."""
    demo_blocks = []
    for i, example in enumerate(five_shot_examples, 1):
        demo_blocks.append(
            f"### Example {i}:\nText: {example['input']}\nEmotions: {example['output']}"
        )

    demonstrations_text = "\n\n".join(demo_blocks)

    return (
        f"{INSTRUCTION_QWEN}\n\n"
        f"{demonstrations_text}\n\n"
        f"### New Text:\n"
        f"Text: {json_data['input']}\n"
        f"Emotions:"
    )


def prepare_prompt_zero_shot(json_data: dict[Hashable, Any]) -> str:
    """Formats the model prompt for zero-shot evaluation."""
    return f"{INSTRUCTION_QWEN}\n\n### New Text:\nText: {json_data['input']}\nEmotions:"
