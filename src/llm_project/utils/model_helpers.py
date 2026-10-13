import torch
from transformers import AutoTokenizer


def load_tokenizer(
    model_name: str,
    padding_side: str,
    trust_remote_code: bool = False,
):
    """Load and configure a tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = padding_side

    if tokenizer.chat_template is None:
        raise ValueError(f"{model_name} does not provide a chat template.")

    return tokenizer


def sync_special_tokens(
    model,
    tokenizer,
) -> None:
    """Synchronize tokenizer and model special tokens."""

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id


@torch.inference_mode()
def generate_texts(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    do_sample: bool = False,
) -> list[str]:
    """Generate model responses for multiple prompts."""

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
            pad_token_id=(tokenizer.pad_token_id),
            eos_token_id=(tokenizer.eos_token_id),
        )

        generated = generated[:, inputs["input_ids"].shape[1] :]

        outputs.extend(
            tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )
        )

    return outputs
