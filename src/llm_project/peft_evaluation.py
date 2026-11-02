from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm_project.metrics import (
    compute_metrics,
    confusion_matrices,
    gold_labels,
    parse_prediction,
)
from llm_project.utils import load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_evaluation_paths(language: str, condition: str, split: str) -> dict[str, Path]:
    # target-only adapters are evaluated only on the target language
    data_dir = PROJECT_ROOT / "data" / "processed" / language
    if condition == "target":
        return {language: data_dir / f"{split}_{language}.jsonl"}
    return {
        "en": data_dir / f"{split}_en.jsonl",
        language: data_dir / f"{split}_{language}.jsonl",
    }


def experiment_name(config: dict[str, Any], language: str, condition: str, eval_lang: str) -> str:
    model_name = config["model"]["name"].split("/")[-1]
    experiment = config["experiment"]["name"]
    return f"{model_name}__{language}__{eval_lang}__peft-{condition}__{experiment}"


def create_prompt(example: dict[str, Any]) -> str:
    return f"{example['instruction']}\n\nText:\n{example['input']}"


def load_base_model(config: dict[str, Any]):
    # load the original base model without creating new trainable adapters
    dtype = torch.bfloat16 if config["training"]["bf16"] else torch.float32
    model_name = config["model"]["name"]
    method = config["peft"]["method"]

    if method == "qlora":
        quant = config["peft"].get("quantization", {})
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.get("quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant.get("double_quant", True),
            bnb_4bit_compute_dtype=dtype,
        )
        return AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, quantization_config=bnb_config, device_map="auto"
        )
    if method == "lora":
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map="auto")
    raise ValueError(f"Unknown peft.method: {method!r}")


def load_trained_adapter(model_dir: Path, config: dict[str, Any]):
    # use the tokenizer saved with the adapter to preserve its special tokens
    adapter_dir = model_dir / "adapter"
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = load_base_model(config)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def check_saved_adapter(model_dir: Path) -> list[str]:
    # verify that the adapter is the q_proj/v_proj experiment
    adapter_dir = model_dir / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"No trained adapter at {adapter_dir}; train it first")

    path = adapter_dir / "adapter_config.json"
    adapter_config = json.loads(path.read_text(encoding="utf-8"))
    targets = sorted(adapter_config.get("target_modules") or [])
    if targets != ["q_proj", "v_proj"]:
        raise ValueError(f"Adapter targets {targets}, expected ['q_proj', 'v_proj']")
    return targets


def output_paths(split: str, run: str) -> tuple[Path, Path]:
    # use the same directories and filenames as the existing results pipeline
    prediction = PROJECT_ROOT / "outputs" / "predictions" / "peft" / split / f"{run}.jsonl"
    metrics = PROJECT_ROOT / "outputs" / "metrics" / "peft" / split / f"{run}.json"
    return prediction, metrics


def check_existing_outputs(
    config: dict[str, Any], language: str, condition: str, split: str, overwrite: bool
) -> None:
    # check before loading a large model to avoid an unnecessary GPU job
    if overwrite:
        return
    for eval_lang in get_evaluation_paths(language, condition, split):
        run = experiment_name(config, language, condition, eval_lang)
        for path in output_paths(split, run):
            if path.exists():
                raise FileExistsError(
                    f"Results already exist at {path}\n"
                    "Use --overwrite only when intentionally rerunning this experiment"
                )


@torch.inference_mode()
def evaluate_dataset(
    model,
    tokenizer,
    dataset: Dataset,
    config: dict[str, Any],
    predictions_path: Path,
) -> dict[str, Any]:
    # generate model responses and keep the raw text for error analysis
    tokenizer.padding_side = "left"
    model.config.use_cache = True
    model.eval()

    records: list[dict[str, Any]] = []
    parsed_outputs = []
    batch_size = config["generation"]["batch_size"]

    for start in range(0, len(dataset), batch_size):
        batch = [
            cast(dict[str, Any], dataset[i])
            for i in range(start, min(start + batch_size, len(dataset)))
        ]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": create_prompt(example)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for example in batch
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config["training"]["max_seq_length"],
            add_special_tokens=False,
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        generated = model.generate(
            **inputs,
            max_new_tokens=config["generation"]["max_new_tokens"],
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        answer_tokens = generated[:, inputs["input_ids"].shape[1] :]
        raw_answers = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)

        for example, raw in zip(batch, raw_answers, strict=True):
            parsed = parse_prediction(raw)
            parsed_outputs.append(parsed)
            records.append(
                {
                    "alignment_id": example["alignment_id"],
                    "text": example["input"],
                    "gold": gold_labels(example["output"]),
                    "pred": list(parsed.labels),
                    "raw_output": raw,
                    "malformed": parsed.malformed,
                }
            )

    if not records:
        raise ValueError("Cannot evaluate an empty dataset")

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    # share exactly the same scoring rules with prompting
    gold = [record["gold"] for record in records]
    pred = [record["pred"] for record in records]
    metrics = compute_metrics(gold, pred, parsed_outputs)
    metrics["confusion"] = confusion_matrices(gold, pred)
    return metrics


def evaluate_model(
    model,
    tokenizer,
    language: str,
    condition: str,
    config: dict[str, Any],
    split: str = "dev",
) -> dict[str, Any]:
    # evaluate each required language and save metrics independently
    all_metrics: dict[str, Any] = {}

    for eval_lang, path in get_evaluation_paths(language, condition, split).items():
        run = experiment_name(config, language, condition, eval_lang)
        prediction_path, metrics_path = output_paths(split, run)

        print(f"Evaluating {condition} adapter on {eval_lang}: {path}")
        dataset = load_jsonl(path)

        start = time.perf_counter()
        scores = evaluate_dataset(model, tokenizer, dataset, config, prediction_path)
        scores.update(
            {
                "run": run,
                "split": split,
                "target_language": language,
                "evaluation_language": eval_lang,
                "model": config["model"]["name"],
                "method": "peft",
                "training": condition,
                "random_seed": config["experiment"]["seed"],
                "inference_seconds": round(time.perf_counter() - start, 3),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            }
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
        all_metrics[eval_lang] = scores
        print(f"{run}: micro_f1={scores['micro_f1']:.3f} jaccard={scores['sample_jaccard']:.3f}")

    return all_metrics


def run_peft_evaluation(
    language: str,
    condition: str,
    split: str,
    experiment: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    # load the configuration saved during training, not today's default settings
    if experiment is None:
        config_path = PROJECT_ROOT / "configs" / "peft.yaml"
        default_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        experiment = default_config["experiment"]["name"]

    model_dir = (
        PROJECT_ROOT
        / "outputs"
        / "models"
        / language
        / condition
        / ("" if experiment is None else experiment)
    )
    targets = check_saved_adapter(model_dir)
    config_path = model_dir / "config_used.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    check_existing_outputs(config, language, condition, split, overwrite)
    print(f"Model dir: {model_dir}")
    print(f"Split:     {split}")
    print(f"Method:    {config['peft']['method']}")
    print(f"Targets:   {targets}")
    print(f"GPU:       {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    model, tokenizer = load_trained_adapter(model_dir, config)
    results = evaluate_model(model, tokenizer, language, condition, config, split)
    print("\nMetrics:")
    print(json.dumps(results, indent=2)[:2000])
    return results
