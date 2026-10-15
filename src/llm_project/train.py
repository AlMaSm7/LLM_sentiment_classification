import json
import platform
from pathlib import Path
from typing import cast

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer  # type: ignore

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42
EPOCHS = 2
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
MAX_LENGTH = 512
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
USE_4BIT = True
USE_WANDB = False
EMOTIONS = (
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
)

TRAIN_SYSTEM_PROMPT = (
    "Identify all emotions expressed in the text. "
    "Return only a JSON list using one or more of these labels:\n"
    "anger, anticipation, disgust, fear, joy, sadness, surprise, trust."
)
TRAIN_USER_PROMPT = "Subtitle: {text}\nnEmotion labels (JSON):"


def read_rows(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run prepare.py first")
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"No training or development examples in {path}")
    return rows


def format_examples(rows, condition):
    data = []
    for row in rows:
        languages = ("en", "target") if condition == "bilingual" else (condition,)
        for language in languages:
            data.append(
                {
                    "prompt": [
                        {"role": "system", "content": TRAIN_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": TRAIN_USER_PROMPT.format(text=row[language]),
                        },
                    ],
                    "completion": [{"role": "assistant", "content": json.dumps(row["labels"])}],
                }
            )
    return Dataset.from_list(data)


def train(lang, condition):
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU")

    set_seed(SEED)

    folder = Path("data/processed") / lang
    training = format_examples(read_rows(folder / "train.jsonl"), condition)
    development = format_examples(read_rows(folder / "dev.jsonl"), condition)
    output = Path("outputs/models") / lang / condition
    if (output / "adapter").exists():
        raise FileExistsError(
            f"Adapter already exists at {output / 'adapter'}; move it before retraining"
        )
    output.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        if USE_4BIT
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        dtype=dtype,
        quantization_config=quantization,
    )
    # align model special tokens with the tokenizer
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if model.generation_config is not None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    model.config.use_cache = False

    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM",
    )
    settings = SFTConfig(
        output_dir=str(output / "checkpoints"),
        max_length=MAX_LENGTH,
        completion_only_loss=True,
        packing=False,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        gradient_checkpointing=True,
        logging_steps=20,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        seed=SEED,
        report_to="wandb" if USE_WANDB else "none",
    )
    trainer = SFTTrainer(
        model=model,
        args=settings,
        train_dataset=training,
        eval_dataset=development,
        peft_config=lora,
        processing_class=tokenizer,
    )
    training_output = trainer.train()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")

    model_for_stats = cast(PeftModel, trainer.model)
    trainable, total = model_for_stats.get_nb_trainable_parameters()
    metadata = {
        "model": MODEL,
        "language": lang,
        "condition": condition,
        "seed": SEED,
        "training_examples": len(training),
        "development_examples": len(development),
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "max_length": MAX_LENGTH,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGETS,
        "quantization": "4bit nf4 double_quant" if USE_4BIT else "none",
        "checkpoint_rule": "minimum development loss",
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_dev_loss": trainer.state.best_metric,
        "train_system_prompt": TRAIN_SYSTEM_PROMPT,
        "train_runtime_seconds": training_output.metrics.get("train_runtime"),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    (output / "settings.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved best adapter to {output / 'adapter'}")
