from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import Dataset

from llm_project.constants import INSTRUCTION

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@torch.inference_mode()
def predict(
    model,
    tokenizer,
    text: str,
    config: dict[str, Any],
) -> tuple[str, str]:
    """Predict emotions for one sentence and return both prediction and prompt."""

    message = f"{INSTRUCTION}\n\nText:\n{text}"

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
    )

    tokenizer.padding_side = "left"
    model.config.use_cache = True
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generated = model.generate(
        **inputs,
        max_new_tokens=config["generation"]["max_new_tokens"],
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    answer_tokens = generated[:, inputs["input_ids"].shape[1] :]
    prediction = tokenizer.decode(answer_tokens[0], skip_special_tokens=True)

    return prediction, prompt


def run_inference(
    model,
    tokenizer,
    texts: list[str],
    language: str,
    condition: str,
    config: dict[str, Any],
    dataset: Dataset | None = None,
) -> list[dict[str, str]]:
    """Run inference on sample texts and optionally a full test dataset for error analysis."""

    # Process sample texts for terminal printout and W&B logging
    text_results = []
    for text in texts:
        prediction, _ = predict(model, tokenizer, text, config)
        text_results.append(
            {
                "text": text,
                "prediction": prediction,
            }
        )

    # Process the full dataset for error analysis in notebook
    if dataset is not None:
        dataset_results = []
        for row in dataset:
            text = row["input"]
            prediction, prompt = predict(model, tokenizer, text, config)

            dataset_results.append(
                {
                    "alignment_id": row.get("alignment_id", ""),
                    "input_text": text,
                    "input_prompt": prompt,
                    "prediction": prediction,
                    "true_labels": row.get("output", ""),
                }
            )

        print(f"Saving test dataset predictions for {language} - {condition}...")
        
        prediction_dir = (
            PROJECT_ROOT
            / "outputs"
            / "predictions"
            / "peft_analysis"
            / language
            / condition
            / config["experiment"]["name"]
        )
        prediction_dir.mkdir(parents=True, exist_ok=True)

        time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = prediction_dir / f"test_inference_{time_stamp}.csv"

        df = pd.DataFrame(dataset_results)
        df.to_csv(output_path, index=False, encoding="utf-8")

        print(f"Test dataset predictions saved: {output_path}")

    return text_results
