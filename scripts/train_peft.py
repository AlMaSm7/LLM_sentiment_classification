import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from llm_project.peft_evaluation import (
    compute_metrics,
    labels_to_binary,
    parse_prediction,
)
from llm_project.peft_training import (
    load_jsonl,
    resolve_dataset_paths,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument(
        "--condition",
        choices=["english", "target", "bilingual"],
        required=True,
    )
    return parser.parse_args()

#text that goes into the encoder: instruction + subtitle line
def make_source(row: dict) -> str:
    return f"{row['instruction']}\n\nText: {row['input']}"


#turns a dataframe into token ID
def prepare(df, tokenizer, max_src: int, max_tgt: int) -> Dataset:
    ds = Dataset.from_dict(
        {
            "source": [make_source(r) for r in df.to_dict("records")],
            "target": df["output"].tolist(),
        }
    )
    #text -> subword pieces -> integer IDs
    #each ID is later looked up in the embedding table inside the model
    def fn(batch):
        enc = tokenizer(batch["source"], max_length=max_src, truncation=True)
        enc["labels"] = tokenizer(
            text_target=batch["target"], max_length=max_tgt, truncation=True
        )["input_ids"]
        return enc

    return ds.map(fn, batched=True, remove_columns=["source", "target"])


def main() -> None:
    args = parse_args()
    #read the YAML
    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    p = config["peft"]
    t = config["training"]
    g = config["generation"]
    seed = config["experiment"]["seed"]
    max_src = t["max_seq_length"]
    max_tgt = g["max_new_tokens"]
    model_name = config["model"]["name"]

    train_file, _, _ = resolve_dataset_paths(
        PROJECT_ROOT, args.language, args.condition
    )
    train_data = load_jsonl(train_file)
    
    # One folder per experiment / language / condition, so runs never overwrite.
    out_dir = (
    PROJECT_ROOT / config["output"]["root_dir"]
    / config["experiment"]["name"] / f"{args.language}_{args.condition}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Language: {args.language}")
    print(f"Condition: {args.condition}")
    print(f"Train: {train_file} ({len(train_data):,} examples)")
    print(f"Model: {model_name}")

    # ---- model + LoRA ----
    #tokenizer, sentencepiece subword tokenizer that mt0 was pretrained with
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Pretrained mT0: embeddings + encoder stack + decoder stack
    # (attention, feed-forward, residuals, layer norm inside).
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, attn_implementation="eager")
    # LoRA: freeze the original weights W and add small trainable matrices A, B
    # to the target modules, so the effective weight is W + (alpha/r) * B @ A.
    model = get_peft_model(
        model,
        LoraConfig(
            r=p["rank"],
            lora_alpha=p["alpha"],
            lora_dropout=p["dropout"],
            use_dora=p.get("use_dora", False),
            target_modules=p["target_modules"],
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        ),
    )
    model.print_trainable_parameters()

    # ---- train ----
    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=str(out_dir),
            learning_rate=t["learning_rate"], #step size
            num_train_epochs=t["epochs"], #passes over the training data
            per_device_train_batch_size=t["batch_size"], #examples per update
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            seed=seed, #controls data shufflings
            save_strategy="no",
            logging_steps=50,
            report_to="none",
        ),
        train_dataset=prepare(train_data, tokenizer, max_src, max_tgt),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=-100
        ),
    )
    trainer.train()

    trainer.model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(config))

    # ---- evaluate on dev, once per language ----
    model.eval()
    results = {}
    data_dir = PROJECT_ROOT / "data" / "processed" / args.language

    for suffix in ["en", args.language]:
        eval_df = load_jsonl(data_dir / f"dev_{suffix}.jsonl")
        sources = [make_source(r) for r in eval_df.to_dict("records")]

        preds = []
        for i in range(0, len(sources), 32):
            batch = tokenizer(
                sources[i : i + 32],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_src,
            ).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **batch,
                    max_new_tokens=max_tgt,
                    do_sample=g["do_sample"],
                    num_beams=g.get("num_beams", 1),
                )
            preds += tokenizer.batch_decode(gen, skip_special_tokens=True)

        y_pred = labels_to_binary(parse_prediction(x) for x in preds)
        y_true = labels_to_binary(parse_prediction(x) for x in eval_df["output"])
        results[suffix] = compute_metrics(y_true, y_pred)

        pred_file = out_dir / f"dev_predictions_{suffix}.jsonl"
        with pred_file.open("w", encoding="utf-8") as f:
            for row, pred in zip(eval_df.to_dict("records"), preds):
                f.write(
                    json.dumps({**row, "prediction": pred}, ensure_ascii=False) + "\n"
                )

    (out_dir / "dev_metrics.json").write_text(json.dumps(results, indent=2))
    print({k: round(v["macro_f1"], 4) for k, v in results.items()})


if __name__ == "__main__":
    main()
