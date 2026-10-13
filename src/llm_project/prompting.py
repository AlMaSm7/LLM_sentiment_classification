from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
)

from llm_project.evaluation import (
    evaluate_outputs,
)
from llm_project.utils.data_io import get_processed_path, load_jsonl_records, save_json
from llm_project.utils.model_helpers import generate_texts, load_tokenizer, sync_special_tokens
from llm_project.utils.prompts import build_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_model(
    model_key: str,
    config: dict[str, Any],
):
    """Load one prompting model."""

    model_config = config["models"][model_key]

    model_name = model_config["name"]

    trust_remote_code = model_config.get(
        "trust_remote_code",
        False,
    )

    tokenizer = load_tokenizer(
        model_name=model_name,
        padding_side="left",
        trust_remote_code=(trust_remote_code),
    )

    if config["generation"]["bf16"]:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=(trust_remote_code),
        device_map={"": torch.cuda.current_device()} if torch.cuda.is_available() else None,
    )

    sync_special_tokens(
        model,
        tokenizer,
    )

    model.eval()

    return model, tokenizer


def get_instruction(
    config: dict[str, Any],
    prompt_version: str,
) -> str:
    """Return one prompt wording."""

    variants = config["prompt"]["variants"]

    if prompt_version not in variants:
        raise ValueError(f"Unknown prompt version: {prompt_version}")

    return variants[prompt_version].strip()


def load_few_shot_examples(
    language: str,
    eval_language: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Load the five fixed training demonstrations."""

    demo_ids = [str(alignment_id) for alignment_id in config["few_shot"][language]]

    variant = "en" if eval_language == "en" else language

    path = get_processed_path(
        language=language,
        split="train",
        variant=variant,
    )

    rows = load_jsonl_records(path)

    rows_by_id = {str(row["alignment_id"]): row for row in rows}

    missing_ids = [alignment_id for alignment_id in demo_ids if alignment_id not in rows_by_id]

    if missing_ids:
        raise ValueError(f"Missing five-shot alignment IDs: {missing_ids}")

    return [rows_by_id[alignment_id] for alignment_id in demo_ids]


def build_five_shot_prompt(
    text: str,
    instruction: str,
    examples: list[dict[str, Any]],
) -> str:
    """Build a prompt containing five demonstrations."""

    parts = [
        instruction,
        "Examples:",
    ]

    for index, example in enumerate(
        examples,
        start=1,
    ):
        parts.append(f"Example {index}\nText:\n{example['input']}\nOutput:\n{example['output']}")

    parts.append(f"Now classify the following text.\nText:\n{text}")

    return "\n\n".join(parts)


def evaluate_prompting(
    model,
    tokenizer,
    model_key: str,
    language: str,
    method: str,
    split: str,
    prompt_version: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one prompting experiment."""

    if method not in {
        "zero_shot",
        "five_shot",
    }:
        raise ValueError(f"Unknown prompting method: {method}")

    instruction = get_instruction(
        config,
        prompt_version,
    )

    base_output_dir = PROJECT_ROOT / "outputs"

    metrics_dir = (
        base_output_dir / "metrics" / "prompting" / language / model_key / method / prompt_version
    )

    predictions_dir = (
        base_output_dir
        / "predictions"
        / "prompting"
        / language
        / model_key
        / method
        / prompt_version
    )

    all_metrics = {}
    prompt_templates = {}

    eval_languages = [
        "en",
        language,
    ]

    for eval_language in eval_languages:
        path = get_processed_path(
            language=language,
            split=split,
            variant=eval_language,
        )

        rows = load_jsonl_records(path)

        examples = []

        if method == "five_shot":
            examples = load_few_shot_examples(
                language=language,
                eval_language=(eval_language),
                config=config,
            )

        if method == "zero_shot":
            prompts = [
                build_prompt(
                    text=row["input"],
                    instruction=instruction,
                )
                for row in rows
            ]

            prompt_templates[eval_language] = build_prompt(
                text="<UTTERANCE>",
                instruction=instruction,
            )

        else:
            prompts = [
                build_five_shot_prompt(
                    text=row["input"],
                    instruction=instruction,
                    examples=examples,
                )
                for row in rows
            ]

            prompt_templates[eval_language] = build_five_shot_prompt(
                text="<UTTERANCE>",
                instruction=instruction,
                examples=examples,
            )

        print(f"Evaluating {eval_language}")

        raw_outputs = generate_texts(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=config["generation"]["batch_size"],
            max_length=config["generation"]["max_seq_length"],
            max_new_tokens=config["generation"]["max_new_tokens"],
            do_sample=config["generation"]["do_sample"],
        )

        metrics, predictions = evaluate_outputs(
            rows=rows,
            raw_outputs=raw_outputs,
        )

        all_metrics[eval_language] = metrics

        save_json(
            predictions,
            predictions_dir / (f"{split}_{eval_language}.json"),
        )

    save_json(
        all_metrics,
        metrics_dir / f"{split}_metrics.json",
    )

    prompt_details = {
        "model": (config["models"][model_key]["name"]),
        "method": method,
        "prompt_version": (prompt_version),
        "instruction": instruction,
        "generation": (config["generation"]),
        "demonstration_ids": (config["few_shot"][language] if method == "five_shot" else []),
        "demonstration_selection": (
            config["few_shot"]["selection_procedure"] if method == "five_shot" else None
        ),
        "prompt_templates": (prompt_templates),
    }

    save_json(
        prompt_details,
        metrics_dir / "prompt_details.json",
    )

    return all_metrics
