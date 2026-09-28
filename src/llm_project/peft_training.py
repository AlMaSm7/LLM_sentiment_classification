import json
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from llm_project.utils import load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict[str, Any]:
    """Load the PEFT config."""

    config_path = PROJECT_ROOT / "configs" / "peft.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_training_paths(
    language: str,
    condition: str,
) -> tuple[Path, Path]:
    """Find the correct train and dev files."""

    if condition == "english":
        suffix = "en"
    elif condition == "target":
        suffix = language
    elif condition == "bilingual":
        suffix = "bilingual"
    else:
        raise ValueError(f"Unknown condition: {condition}")

    data_dir = PROJECT_ROOT / "data" / "processed" / language

    train_path = data_dir / f"train_{suffix}.jsonl"
    dev_path = data_dir / f"dev_{suffix}.jsonl"

    return train_path, dev_path


def load_training_data(
    language: str,
    condition: str,
) -> tuple[Dataset, Dataset]:
    """Load raw train and development datasets."""

    train_path, dev_path = get_training_paths(
        language,
        condition,
    )

    print(f"Train data: {train_path}")
    print(f"Dev data:   {dev_path}")

    return (
        load_jsonl(train_path),
        load_jsonl(dev_path),
    )


def build_model(
    config: dict[str, Any],
):
    """Load Qwen and attach LoRA adapters."""

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # right padding is fine during training.
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if config["training"]["bf16"] else torch.float32

    match config["peft"]["method"]:
        case "lora":
            model = AutoModelForCausalLM.from_pretrained(
                config["model"]["name"],
                dtype=dtype,
            )
        case "qlora":
            # TODO: Add configurations to yaml
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            model = AutoModelForCausalLM.from_pretrained(
                config["model"]["name"], dtype=dtype, quantization_config=bnb_config
            )
            model = prepare_model_for_kbit_training(model)
        case _:
            model = AutoModelForCausalLM.from_pretrained(
                config["model"]["name"],
                dtype=dtype,
            )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["peft"]["rank"],
        lora_alpha=config["peft"]["alpha"],
        lora_dropout=config["peft"]["dropout"],
        target_modules=config["peft"]["target_modules"],
        bias="none",
    )
    model = get_peft_model(
        model,
        lora_config,
    )

    model.config.use_cache = False

    model.print_trainable_parameters()

    return model, tokenizer


def create_user_message(
    example: dict[str, Any],
) -> str:
    """Create the user message used during training."""

    return f"{example['instruction']}\n\nText:\n{example['input']}"


def prepare_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int,
) -> Dataset:
    """Tokenize the dataset and mask the prompt."""

    def tokenize(
        example: dict[str, Any],
    ) -> dict[str, Any]:

        # user chat message:
        user = {
            "role": "user",
            "content": create_user_message(example),
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
            [user, assistant],
            tokenize=False,
            add_generation_prompt=False,  # the assistant shouldn't answer
        )

        prompt_tokens = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )

        full_tokens = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,  # cut off sequence if it exceeds max_length
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
            "attention_mask": attention_mask,
            "labels": labels,
            "has_answer": any(
                label != -100 for label in labels
            ),  # e.g. the prompt uses all tokens and cuts off before the answer -> answer not included
        }

    dataset = dataset.map(
        tokenize,
        remove_columns=dataset.column_names,  # remove original columns after tokenizing
    )

    dataset = dataset.filter(lambda example: example["has_answer"])

    return dataset.remove_columns(["has_answer"])


def get_model_dir(
    language: str,
    condition: str,
    config: dict[str, Any],
) -> Path:
    """Return the output directory for the trained model."""

    return PROJECT_ROOT / "outputs" / "models" / language / condition / config["experiment"]["name"]


def train_model(
    model,
    tokenizer,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    config: dict[str, Any],
    model_dir: Path,
    experiment: str,
) -> Trainer:
    """Train the LoRA adapter."""

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        learning_rate=config["training"]["learning_rate"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["eval"]["batch_size"],
        gradient_accumulation_steps=(config["training"]["gradient_accumulation_steps"]),
        num_train_epochs=config["training"]["epochs"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # false since we are using eval_loss as metric for training
        bf16=config["training"]["bf16"],
        seed=config["experiment"]["seed"],
        save_total_limit=2,  # keep two checkpoints in total
        report_to="wandb" if config["use_wandb"] else None,
        run_name=experiment,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
        ),
    )

    trainer.train()

    return trainer


def save_model(
    trainer: Trainer,
    tokenizer,
    config: dict[str, Any],
    model_dir: Path,
    language: str,
    condition: str,
) -> None:
    """Save adapter, tokenizer, config and metadata."""

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

    metadata = {
        "language": language,
        "condition": condition,
        "seed": config["experiment"]["seed"],
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "hardware": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
    }

    with (model_dir / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )
