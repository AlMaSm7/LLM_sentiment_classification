from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from llm_project.utils import (
    PROJECT_ROOT,
    build_prompt,
    get_processed_path,
    load_jsonl_dataset,
    save_json,
)


def get_training_paths(
    language: str,
    condition: str,
) -> tuple[Path, Path]:
    """Return train and development paths."""

    match condition:
        case "english":
            variant = "en"

        case "target":
            variant = language

        case "bilingual":
            variant = "bilingual"

        case _:
            raise ValueError(f"Unknown condition: {condition}")

    train_path = get_processed_path(
        language=language,
        split="train",
        variant=variant,
    )

    dev_path = get_processed_path(
        language=language,
        split="dev",
        variant=variant,
    )

    return train_path, dev_path


def load_training_data(
    language: str,
    condition: str,
) -> tuple[Dataset, Dataset]:
    """Load the PEFT training and development data."""

    train_path, dev_path = get_training_paths(
        language,
        condition,
    )

    print(f"Train data: {train_path}")
    print(f"Dev data:   {dev_path}")

    return (
        load_jsonl_dataset(train_path),
        load_jsonl_dataset(dev_path),
    )


def build_quantization_config(
    config: dict[str, Any],
    dtype: torch.dtype,
) -> BitsAndBytesConfig:
    """Create the configured QLoRA quantization setup."""

    quantization = config["peft"]["quantization"]

    bits = quantization["bits"]

    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=(quantization["quant_type"]),
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=(quantization["double_quant"]),
        )

    if bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=(quantization["llm_int8_threshold"]),
            llm_int8_has_fp16_weight=(quantization["llm_int8_has_fp16_weight"]),
        )

    raise ValueError("Quantization bits must be 4 or 8.")


def load_base_model(
    config: dict[str, Any],
    for_training: bool,
):
    """Load the base model for LoRA or QLoRA."""

    model_name = config["model"]["name"]

    if config["training"]["bf16"]:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    method = config["peft"]["method"]

    match method:
        case "lora":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                device_map={"": (torch.cuda.current_device())}
                if torch.cuda.is_available()
                else None,
            )

            # if torch.cuda.is_available():
            #     model = model.cuda()

        case "qlora":
            bnb_config = build_quantization_config(
                config,
                dtype,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                quantization_config=(bnb_config),
                device_map={"": (torch.cuda.current_device())}
                if torch.cuda.is_available()
                else None,
            )

            if for_training:
                model = prepare_model_for_kbit_training(
                    model,
                    gradient_checkpointing_kwargs={
                        "use_reentrant": False,
                    },
                )

        case _:
            raise ValueError(f"Unknown PEFT method: {method}")

    return model


def load_tokenizer(
    model_name: str,
    padding_side: str,
):
    """Load and configure the tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)

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


def build_model(
    config: dict[str, Any],
):
    """Build a trainable LoRA or QLoRA model."""

    model_name = config["model"]["name"]

    tokenizer = load_tokenizer(
        model_name=model_name,
        padding_side="right",
    )

    model = load_base_model(
        config=config,
        for_training=True,
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

    sync_special_tokens(
        model,
        tokenizer,
    )

    model.config.use_cache = False

    model.print_trainable_parameters()

    return model, tokenizer


def prepare_dataset(
    dataset: Dataset,
    tokenizer,
    config: dict[str, Any],
) -> Dataset:
    """Tokenize examples and mask prompt tokens."""

    max_length = config["training"]["max_seq_length"]

    def tokenize(
        example: dict[str, Any],
    ) -> dict[str, Any]:

        # user chat message:
        user = {
            "role": "user",
            "content": build_prompt(
                text=example["input"],
                instruction=(example["instruction"]),
            ),
        }

        # what the model should produce:
        assistant = {
            "role": "assistant",
            "content": example["output"],
        }

        # user prompt without the correct answer
        prompt_text = tokenizer.apply_chat_template(
            [user],
            tokenize=False,
            add_generation_prompt=True,  # the assistant should answer
        )

        # full conversation including the correct answer to give the model for training
        full_text = tokenizer.apply_chat_template(
            [
                user,
                assistant,
            ],
            tokenize=False,
            add_generation_prompt=False,  # the assistant shouldn't answer
        )

        prompt_tokens = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,  # cut off sequence if it exceeds max_length
            max_length=max_length,
        )

        full_tokens = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )

        input_ids = list(full_tokens["input_ids"])

        # check for padded tokens
        attention_mask = list(full_tokens["attention_mask"])

        labels = input_ids.copy()

        prompt_length = min(
            len(prompt_tokens["input_ids"]),
            len(labels),
        )

        # -100 means not to calculate loss for the user's prompt
        labels[:prompt_length] = [-100] * prompt_length

        return {
            "input_ids": input_ids,
            "attention_mask": (attention_mask),
            "labels": labels,
            "has_answer": any(label != -100 for label in labels),
            # e.g. the prompt uses all tokens and cuts off before the answer -> answer not included
        }

    dataset = dataset.map(
        tokenize,
        remove_columns=(dataset.column_names),
    )

    dataset = dataset.filter(lambda example: example["has_answer"])

    return dataset.remove_columns(["has_answer"])


def get_model_dir(
    language: str,
    condition: str,
    config: dict[str, Any],
) -> Path:
    """Return the model output directory."""

    return PROJECT_ROOT / "outputs" / "models" / language / condition / config["experiment"]["name"]


def train_model(
    model,
    tokenizer,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    language: str,
    condition: str,
    config: dict[str, Any],
) -> Trainer:
    """Train the PEFT model."""

    model_dir = get_model_dir(
        language,
        condition,
        config,
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_name = f"{language}_{condition}_{config['experiment']['name']}"

    training_args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        learning_rate=(config["training"]["learning_rate"]),
        per_device_train_batch_size=(config["training"]["batch_size"]),
        per_device_eval_batch_size=(config["training"]["batch_size"]),
        gradient_accumulation_steps=(config["training"]["gradient_accumulation_steps"]),
        num_train_epochs=(config["training"]["epochs"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=("eval_loss"),
        greater_is_better=False,
        bf16=(config["training"]["bf16"]),
        seed=(config["experiment"]["seed"]),
        save_total_limit=(config["training"]["save_total_limit"]),
        logging_steps=(config["training"]["logging_steps"]),
        report_to=("wandb" if config["wandb"]["enabled"] else "none"),
        run_name=run_name,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()

    return trainer


def save_model(
    trainer: Trainer,
    tokenizer,
    language: str,
    condition: str,
    config: dict[str, Any],
) -> None:
    """Save the selected adapter and metadata."""

    model_dir = get_model_dir(
        language,
        condition,
        config,
    )

    adapter_dir = model_dir / "adapter"

    trainer.save_model(str(adapter_dir))

    tokenizer.save_pretrained(adapter_dir)

    with (model_dir / "config_used.yaml").open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )

    hardware = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    metadata = {
        "language": language,
        "condition": condition,
        "model": (config["model"]["name"]),
        "peft_method": (config["peft"]["method"]),
        "quantization": (
            config["peft"]["quantization"] if (config["peft"]["method"] == "qlora") else None
        ),
        "seed": (config["experiment"]["seed"]),
        "best_checkpoint": (trainer.state.best_model_checkpoint),
        "checkpoint_rule": ("lowest development loss"),
        "hardware": hardware,
    }

    save_json(
        metadata,
        model_dir / "metadata.json",
    )


def load_saved_model(
    language: str,
    condition: str,
    config: dict[str, Any],
):
    """Load a previously trained PEFT adapter."""

    model_dir = get_model_dir(
        language,
        condition,
        config,
    )

    adapter_dir = model_dir / "adapter"

    if not adapter_dir.exists():
        raise FileNotFoundError(adapter_dir)

    base_model = load_base_model(
        config=config,
        for_training=False,
    )

    tokenizer = load_tokenizer(
        model_name=str(adapter_dir),
        padding_side="left",
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
    )

    sync_special_tokens(
        model,
        tokenizer,
    )

    model.eval()

    return model, tokenizer
