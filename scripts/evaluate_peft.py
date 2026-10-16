from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm_project.peft_evaluation import evaluate_model


def load_base_model(config: dict):
    dtype = torch.bfloat16 if config["training"]["bf16"] else torch.float32
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
            config["model"]["name"],
            dtype=dtype,
            quantization_config=bnb_config,
            device_mparser="auto",
        )
    if method == "lora":
        return AutoModelForCausalLM.from_pretrained(
            config["model"]["name"], dtype=dtype, device_mparser="auto"
        )
    raise ValueError(f"Unknown peft.method: {method!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, help="assigned language, e.g. sv")
    parser.add_argument(
        "--condition", required=True, help="training condition, e.g. english or target"
    )
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--experiment", help="experiment name (default: from configs/peft.yaml)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    experiment = args.experimment
    if experiment is None:
        experiment = yaml.safe_load(open(PROJECT_ROOT / "configs" / "peft.yaml"))["experiment"][
            "name"
        ]

    model_dir = PROJECT_ROOT / "outputs" / "models" / args.language / args.condition / experiment
    # this is the small matrices with q_proj and v_proj,
    adapter_dir = model_dir / "adapter"
    if not adapter_dir.exists():
        sys.exit(f"No trained adapter at {adapter_dir}. Train it first (run_peft.py).")

    config = yaml.safe_load(open(model_dir / "config_used.yaml"))
    # Sanity checks: the saved adapter must be the fixed q_proj + v_proj version.
    adapter_cfg = json.load(open(adapter_dir / "adapter_config.json"))
    targets = sorted(adapter_cfg.get("target_modules") or [])
    if targets != ["q_proj", "v_proj"]:
        sys.exit(f"adapter targets {targets}, expected ['q_proj', 'v_proj']. Wrong (old) model?")
    # /-1 splits the hf name
    model_name = config["model"]["name"].split("/")[-1]
    pred_dir = PROJECT_ROOT / "outputs" / "predictions" / "peft_raw" / args.split
    existing = list(pred_dir.glob(f"{model_name}__*__peft-{args.condition}__{experiment}.jsonl"))
    if existing and not args.overwrite:
        sys.exit(
            f"{args.split} predictions already exist for this model: {[p.name for p in existing]}.\n"
            "Each condition is evaluated on test ONCE. Use --overwrite only to redo a crashed run."
        )

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Model:   {model_dir}")
    print(f"Split:   {args.split}")
    print(
        f"Method:  {config['peft']['method']}, quantization: {config['peft'].get('quantization')}"
    )
    print(f"Targets: {targets}")
    print(f"GPU:     {gpu}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(load_base_model(config), adapter_dir)
    model.eval()

    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        language=args.language,
        condition=args.condition,
        config=config,
        split=args.split,
    )
    print("\nMetrics (PEFT code's own scoring; re-score with rescore_predictions.py):")
    print(json.dumps(metrics, indent=2)[:2000])


if __name__ == "__main__":
    main()
