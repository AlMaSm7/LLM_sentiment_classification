from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from llm_project.single_model_prediction.prompt_formatting import (
    check_output_formatting,
    fetch_fixed_five_shot_examples,
    prepare_prompt_five_shot,
    prepare_prompt_zero_shot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_inference_model(model_id: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Loads the tokenizer and model for inference.
    """
    print(f"Loading tokenizer and model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"device_map": "auto"}
    kwargs["dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    return model, tokenizer


def _generate_prediction(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_new_tokens: int = 64,
) -> str:
    """
    Generates a deterministic prediction using the model's native chat template.
    """
    # 1. Wrap your raw prompt string in the expected conversational format
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 2. Apply the model's specific chat template (e.g., ChatML for Qwen)
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # 3. Tokenize the properly formatted prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Enforces reproducible deterministic decoding
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 4. Slice the output to exclude the original prompt tokens
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]

    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

def run_inference(
    config: dict[str, Any],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: pd.DataFrame,
    eval_dataset: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Runs inference on the provided evaluation dataset.
    Requires the train_dataset to fetch fixed demonstrations for the five-shot condition.
    """
    results = []

    method = config["run"]["mode"]

    if method == "five_shot":
        # Extract the fixed five demonstrations from the training set
        five_shot_examples = fetch_fixed_five_shot_examples(
            train_dataset, config["fixed_five_shot_ids"]
        )

        for _, row in eval_dataset.iterrows():
            row = row.to_dict()
            prompt = prepare_prompt_five_shot(row, five_shot_examples)
            prediction = _generate_prediction(prompt, model, tokenizer)

            normalized_prediction = check_output_formatting(prediction)
            results.append(
                {
                    "alignment_id": row["alignment_id"],
                    "input_prompt": prompt,
                    "prediction": prediction,
                    "normalized_prediction": normalized_prediction,
                    "true_labels": row["output"],
                }
            )

    elif method == "zero_shot":
        for _, row in eval_dataset.iterrows():
            row = row.to_dict()
            prompt = prepare_prompt_zero_shot(row)
            prediction = _generate_prediction(prompt, model, tokenizer)

            results.append(
                {
                    "alignment_id": row["alignment_id"],
                    "input_prompt": prompt,
                    "prediction": prediction,
                    "normalized_prediction": check_output_formatting(prediction),
                    "true_labels": row["output"],
                }
            )
    else:
        raise ValueError(f"Unknown prompting mode: {method}")

    return results
