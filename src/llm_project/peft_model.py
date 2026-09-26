"""Model, tokenizer, and LoRA setup."""

from pathlib import Path
from typing import Any

import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def get_dtype(
    config: dict[str, Any],
) -> torch.dtype:
    """Return the configured numerical precision."""

    if config["training"].get(
        "bf16",
        False,
    ):
        return torch.bfloat16

    if config["training"].get(
        "fp16",
        False,
    ):
        return torch.float16

    return torch.float32


def load_tokenizer(
    config: dict[str, Any],
    padding_side: str,
):
    """Load the model tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = padding_side

    if tokenizer.chat_template is None:
        raise ValueError("The selected model needs a chat template.")

    return tokenizer


def build_lora_model(
    config: dict[str, Any],
):
    """Load the base model and attach LoRA adapters."""

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        dtype=get_dtype(config),
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["peft"]["rank"],
        lora_alpha=(config["peft"]["alpha"]),
        lora_dropout=(config["peft"]["dropout"]),
        target_modules=(config["peft"]["target_modules"]),
        bias="none",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.config.use_cache = False

    model.print_trainable_parameters()

    return model


def load_trained_model(
    config: dict[str, Any],
    adapter_path: Path,
):
    """Load the original model together with a trained LoRA adapter."""

    base_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        dtype=get_dtype(config),
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
    )

    model.config.use_cache = True

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    return model
