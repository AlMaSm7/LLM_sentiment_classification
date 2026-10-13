from pathlib import Path
from typing import Any

import torch

from llm_project.evaluation import (
    evaluate_outputs,
)
from llm_project.utils import (
    PROJECT_ROOT,
    build_prompt,
    get_processed_path,
    load_jsonl_records,
    save_json,
)


def get_evaluation_paths(
    language: str,
    condition: str,
    split: str,
) -> dict[str, Path]:
    """Return datasets required for evaluation."""

    if condition == "target":
        return {
            language: get_processed_path(
                language=language,
                split=split,
                variant=language,
            )
        }

    if condition in {
        "english",
        "bilingual",
    }:
        return {
            "en": get_processed_path(
                language=language,
                split=split,
                variant="en",
            ),
            language: get_processed_path(
                language=language,
                split=split,
                variant=language,
            ),
        }

    raise ValueError(f"Unknown condition: {condition}")


@torch.inference_mode()
def generate_texts(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    do_sample: bool,
) -> list[str]:
    """Generate responses for several prompts."""

    tokenizer.padding_side = "left"

    model.config.use_cache = True
    model.eval()

    outputs = []

    for start in range(
        0,
        len(prompts),
        batch_size,
    ):
        batch_prompts = prompts[start : start + batch_size]

        formatted_prompts = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in batch_prompts
        ]

        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )

        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # prompt + generated response
        generated = generated[:, inputs["input_ids"].shape[1] :]

        outputs.extend(
            tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )
        )

    return outputs


def evaluate_model(
    model,
    tokenizer,
    language: str,
    condition: str,
    split: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a model and save metrics and predictions."""

    paths = get_evaluation_paths(
        language=language,
        condition=condition,
        split=split,
    )

    experiment_name = config["experiment"]["name"]

    metrics_dir = PROJECT_ROOT / "outputs" / "metrics" / language / condition / experiment_name
    predictions_dir = (
        PROJECT_ROOT / "outputs" / "predictions" / language / condition / experiment_name
    )

    all_metrics = {}

    for eval_language, path in paths.items():
        print(f"Evaluating: {eval_language}")

        rows = load_jsonl_records(path)

        prompts = [
            build_prompt(
                text=row["input"],
                instruction=row["instruction"],
            )
            if row["instruction"]
            else build_prompt(text=row["input"])
            for row in rows
        ]

        raw_outputs = generate_texts(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=(config["generation"]["batch_size"]),
            max_length=(config["training"]["max_seq_length"]),
            max_new_tokens=(config["generation"]["max_new_tokens"]),
            do_sample=(config["generation"]["do_sample"]),
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

    metrics_path = metrics_dir / f"{split}_metrics.json"

    save_json(
        all_metrics,
        metrics_path,
    )

    print(f"Metrics saved: {metrics_path}")

    return all_metrics
