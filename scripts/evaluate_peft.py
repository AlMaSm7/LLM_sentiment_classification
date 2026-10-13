
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
 
from llm_project.peft_evaluation import evaluate_model  # noqa: E402
 
 
def load_base_model(config: dict):
    """Same base-model loading as build_model in peft_training.py, minus the
    training-only steps (fresh LoRA adapters, prepare_model_for_kbit_training)."""
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
            config["model"]["name"], dtype=dtype, quantization_config=bnb_config,
            device_map="auto",
        )
    if method == "lora":
        return AutoModelForCausalLM.from_pretrained(
            config["model"]["name"], dtype=dtype, device_map="auto"
        )
    raise ValueError(f"Unknown peft.method: {method!r}")
 
 
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, help="assigned language, e.g. sv")
    ap.add_argument("--condition", required=True, help="training condition, e.g. english or target")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--experiment", help="experiment name (default: from configs/peft.yaml)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
 
    experiment = args.experiment or yaml.safe_load(
        (PROJECT_ROOT / "configs" / "peft.yaml").read_text(encoding="utf-8")
    )["experiment"]["name"]
    model_dir = PROJECT_ROOT / "outputs" / "models" / args.language / args.condition / experiment
    adapter_dir = model_dir / "adapter"
    if not adapter_dir.exists():
        sys.exit(f"No trained adapter at {adapter_dir}. Train it first (run_peft.py).")
 
    # The config the model was TRAINED with, not whatever peft.yaml says today.
    config = yaml.safe_load((model_dir / "config_used.yaml").read_text(encoding="utf-8"))
 
    # Sanity checks: the saved adapter must be the fixed q_proj + v_proj version.
    adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    targets = sorted(adapter_cfg.get("target_modules") or [])
    if targets != ["q_proj", "v_proj"]:
        sys.exit(f"Adapter targets {targets}, expected ['q_proj', 'v_proj']. Wrong (old) model?")
 
    model_tag = config["model"]["name"].split("/")[-1]
    pred_dir = PROJECT_ROOT / "outputs" / "predictions" / "peft_raw" / args.split
    existing = list(pred_dir.glob(f"{model_tag}__*__peft-{args.condition}__{experiment}.jsonl"))
    if existing and not args.overwrite:
        sys.exit(
            f"{args.split} predictions already exist for this model: {[p.name for p in existing]}.\n"
            "Each condition is evaluated on test ONCE. Use --overwrite only to redo a crashed run."
        )
 
    print(f"Model dir:  {model_dir}")
    print(f"Split:      {args.split}")
    print(f"Method:     {config['peft']['method']}  quantization={config['peft'].get('quantization')}")
    print(f"Targets:    {targets}")
    print(f"GPU:        {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
 
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    base = load_base_model(config)
    model = PeftModel.from_pretrained(base, adapter_dir)
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