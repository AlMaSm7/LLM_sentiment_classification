import json
from collections.abc import Hashable
from typing import Any

import pandas as pd

from llm_project.constants import EMOTIONS, INSTRUCTION_QWEN


def fetch_fixed_five_shot_examples(
    train_df: pd.DataFrame, fixed_ids: list[str]
) -> list[dict[str, str]]:
    """
    Fetches the five specific training examples corresponding to the provided alignment IDs.
    """
    # Filter the training dataframe to only the specified IDs
    fixed_df = train_df[train_df["alignment_id"].isin(fixed_ids)].copy()

    # Enforce the exact order provided in the YAML list for complete consistency
    fixed_df["id_category"] = pd.Categorical(
        fixed_df["alignment_id"], categories=fixed_ids, ordered=True
    )
    fixed_df = fixed_df.sort_values("id_category")

    demonstrations = []
    for _, row in fixed_df.iterrows():
        # Ensure labels are loaded properly if they are stored as JSON strings
        labels = json.loads(row["output"]) if isinstance(row["output"], str) else row["output"]

        # Serialize back to JSON string format to match the expected prompt structure
        label_string = json.dumps(labels, ensure_ascii=False)

        demonstrations.append({"input": row["input"], "output": label_string})

    return demonstrations


def check_output_formatting(output: str) -> list[str]:
    """
    Parses the JSON output, normalizes it, and filters out duplicate or unpermitted labels.
    Returns an empty list for malformed outputs.
    """
    try:
        parsed_output = json.loads(output)

        if not isinstance(parsed_output, list):
            return []

        if not all(isinstance(emotion, str) for emotion in parsed_output):
            return []

        # Safely build a new list with only normalized, permitted, and unique emotions
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
    """
    Formats the model prompt with the fixed five-shot demonstrations.
    """
    # Build the demonstrations block
    demo_blocks = []
    for i, example in enumerate(five_shot_examples, 1):
        demo_blocks.append(
            f"### Example {i}:\nText: {example['input']}\nEmotions: {example['output']}"
        )

    demonstrations_text = "\n\n".join(demo_blocks)
    
    prompt = (
        f"{INSTRUCTION_QWEN}\n\n"
        f"{demonstrations_text}\n\n"
        f"### New Text:\n"
        f"Text: {json_data['input']}\n"
        f"Emotions:"
    )
    return prompt


def prepare_prompt_zero_shot(json_data: dict[Hashable, Any]) -> str:
    """
    Formats the model prompt for zero-shot evaluation.
    """
    prompt = f"{INSTRUCTION_QWEN}\n### New Text:\nText: {json_data['input']}\nEmotions:"
    return prompt
