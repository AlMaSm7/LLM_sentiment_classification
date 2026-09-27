"""Inference utilities for trained PEFT models."""

import json
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@torch.inference_mode()
def predict(
    model,
    tokenizer,
    text: str,
    config: dict[str, Any],
) -> str:
    """Predict emotions for one sentence."""

    instruction = (
        "Identify all emotions expressed in the text. "
        "Return only a JSON list using one or more of these labels: "
        "anger, anticipation, disgust, fear, joy, sadness, surprise, trust."
    )

    message = f"{instruction}\n\nText:\n{text}"

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": message,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    tokenizer.padding_side = "left"
    model.config.use_cache = True
    model.eval()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    generated = model.generate(
        **inputs,
        max_new_tokens=config["generation"]["max_new_tokens"],
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    answer_tokens = generated[:, inputs["input_ids"].shape[1] :]

    return tokenizer.decode(
        answer_tokens[0],
        skip_special_tokens=True,
    )


def run_inference(
    model,
    tokenizer,
    texts: list[str],
    language: str,
    condition: str,
    config: dict[str, Any],
) -> list[dict[str, str]]:
    """Run several examples and save the predictions."""

    results = []

    for text in texts:
        prediction = predict(
            model,
            tokenizer,
            text,
            config,
        )

        results.append(
            {
                "text": text,
                "prediction": prediction,
            }
        )

    prediction_dir = (
        PROJECT_ROOT
        / "outputs"
        / "predictions"
        / language
        / condition
        / config["experiment"]["name"]
    )

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = prediction_dir / "inference.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Predictions saved: {output_path}")

    return results
