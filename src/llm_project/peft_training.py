import os
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from llm_project.peft_model import build_model_and_tokenizer


def train(
    train_data: pd.DataFrame,
    dev_data: pd.DataFrame,
    config: dict[str, Any],
):
    """Build the configured model and run the selected PEFT method."""
    
    method = config["peft"]["method"].lower()

    if method == "lora":
        model, tokenizer = build_model_and_tokenizer(config)
        return train_lora(model, tokenizer, train_data, dev_data, config)
    if method == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        model, tokenizer = build_model_and_tokenizer(config, bnb_config)
        return train_q_lora(model, tokenizer, train_data, dev_data, config)

    raise ValueError("peft.method must be 'lora' or 'qlora'")

def set_wandb(config: dict[str, Any]):
    if config.get("use_wandb", False):
        os.environ["WANDB_PROJECT"] = config.get("wandb_project", "peft_emotion_classification")
    

def resolve_dataset_paths(
    project_root: Path,
    language: str,
    condition: str,
) -> tuple[Path, Path, Path]:
    """Return the prepared train/dev/test files for one PEFT condition."""
    if condition == "english":
        suffix = "en"
    elif condition == "target":
        suffix = language
    elif condition == "bilingual":
        suffix = "bilingual"
    else:
        raise ValueError("condition must be 'english', 'target', or 'bilingual'")

    data_dir = project_root / "data" / "processed" / language

    return (
        data_dir / f"train_{suffix}.jsonl",
        data_dir / f"dev_{suffix}.jsonl",
        data_dir / f"test_{suffix}.jsonl",
    )


def load_jsonl(path: Path) -> pd.DataFrame:
    """Load one prepared JSONL dataset."""
    return pd.read_json(path, lines=True)


def tokenize_with_chat_template(dictionary: dict, tokenizer) -> dict:
    # chat template for Qwen-7B-Chat (because it uses <|im_start|> and <|im_end|> tags)
    messages = [
        {"role": "system", "content": dictionary.get("instruction", "")},
        {"role": "user", "content": dictionary.get("input", "")},
        {"role": "assistant", "content": dictionary.get("output", "")} 
    ]
    
    # This automatically applies the correct <|im_start|> tags for Qwen
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    tokenized = tokenizer(full_text, truncation=True, max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def tokenize_row(row, tokenizer) -> dict:
    return tokenize_with_chat_template(row, tokenizer)


def prepare_datasets(
    train_data: pd.DataFrame,
    dev_data: pd.DataFrame,
    tokenizer: Any,
    config: dict[str, Any],
) -> tuple[Dataset, Dataset]:
    """Convert, tokenize, and prepare the training and development datasets."""
    hf_train = Dataset.from_pandas(train_data)
    hf_dev = Dataset.from_pandas(dev_data)

    hf_train = hf_train.map(
        lambda row: tokenize_row(row, tokenizer),
        remove_columns=hf_train.column_names,
    )
    hf_dev = hf_dev.map(
        lambda row: tokenize_row(row, tokenizer),
        remove_columns=hf_dev.column_names,
    )

    set_wandb(config)
    return hf_train, hf_dev


def _execute_peft_training(
    model: Any,
    tokenizer: Any,
    train_data: pd.DataFrame,
    dev_data: pd.DataFrame,
    config: dict[str, Any],
    is_qlora: bool = False,
):
    """Shared training loop logic for both LoRA and QLoRA."""
    
    hf_train, hf_dev = prepare_datasets(train_data, dev_data, tokenizer, config)
    
    lora_config = LoraConfig(
        r=config["peft"]["rank"], 
        lora_alpha=config["peft"]["lora_alpha"], 
        target_modules=config["peft"]["target_modules"], 
        lora_dropout=config["peft"]["dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    if is_qlora:
        model = prepare_model_for_kbit_training(model)
        
    # load model
    model = get_peft_model(model, lora_config)
    
    # We want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
        
    # Data collator dynamically pads inputs and labels to max batch length
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )
    
    # sets pad token to eos token if not already set (eos is used for Qwen-7B-Chat, end of sequence)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    output_dir = config["output"]["root_dir"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=5,
        save_steps=100,
        eval_strategy="steps", # Updated from deprecated evaluation_strategy
        eval_steps=100,
        load_best_model_at_end=True,
        report_to="wandb" if config.get("use_wandb", False) else "none",
        run_name=config.get("wandb_run_name", "peft_lora_training"),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train,
        eval_dataset=hf_dev,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    
    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def train_q_lora(
    model: Any,
    tokenizer: Any,
    train_data: pd.DataFrame,
    dev_data: pd.DataFrame,
    config: dict[str, Any],
):
    """Training loop for QLoRA fine-tuning."""
    _execute_peft_training(model, tokenizer, train_data, dev_data, config, is_qlora=True)


def train_lora(
    model: Any,
    tokenizer: Any,
    train_data: pd.DataFrame,
    dev_data: pd.DataFrame,
    config: dict[str, Any],
):
    """Training loop for LoRA fine-tuning."""
    _execute_peft_training(model, tokenizer, train_data, dev_data, config, is_qlora=False)