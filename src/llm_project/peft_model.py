from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def build_model_and_tokenizer(config: dict[str, Any], bnb_config: BitsAndBytesConfig | None = None) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Example placeholder for model/tokenizer loading and LoRA setup."""
    model_id = config["model"]["id"]

    # 1. Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config
    )

    # 2. Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    

    return model, tokenizer

