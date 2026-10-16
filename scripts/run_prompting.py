"""Run Phase 2: {primary, second model} x {English, assigned language} x {zero, five-shot}.

Usage (from the project root, inside the uv environment):
    # 0. choose the five demonstration IDs once, then paste them into the config
    uv run python scripts/run_prompting.py --select-ids

    # 1. main experiment on dev
    uv run python scripts/run_prompting.py --split dev

    # 2. wording-sensitivity analysis (dev only)
    uv run python scripts/run_prompting.py --split dev --sensitivity

    # 3. final test run, ONCE, after the prompts are frozen
    uv run python scripts/run_prompting.py --split test

Add --limit 20 for a quick smoke test.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# `uv sync` normally installs the package; this fallback keeps the script
# working if the project is not installed.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm_project.prompt_evaluation import (  # noqa: E402
    compute_metrics,
    confusion_matrices,
)
from llm_project.prompting import (  # noqa: E402
    DEFAULT_GENERATION,
    DEFAULT_VARIANT,
    INSTRUCTION_VARIANTS,
    build_examples,
    build_prompt,
    generate,
    gold_labels,
    parse_prediction,
    prompt_fingerprint,
    select_covering_ids,
    to_model_input,
)

CONDITIONS = ["zero_shot", "five_shot"]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def data_path(cfg: dict, key: str, lang: str, eval_lang: str | None = None) -> Path:
    data = cfg["data"]
    folder = PROJECT_ROOT / data["dir"].format(lang=lang)
    name = data[key]
    if isinstance(name, dict):  # dev/test: one file per evaluation language
        name = name["en" if eval_lang == "en" else "target"]
    return folder / name.format(lang=lang)


def load_model(name: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=getattr(torch, dtype), device_map="auto"
    )
    model.eval()
    return model, tokenizer


def unload(model) -> None:
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_prompt_card(path: Path, cfg, gen, ids, demos, model_cfg, tokenizer, example_text):
    """Markdown record of the complete prompts, decoding and demo selection."""
    lines = [
        f"# Prompt card: {model_cfg['name']} ({model_cfg['role']})",
        "",
        f"Prompt fingerprint: `{prompt_fingerprint()}`",
        "",
        "## Decoding",
        f"- greedy decoding (do_sample=False), max_new_tokens={gen['max_new_tokens']}",
        f"- batch_size={gen['batch_size']}, left padding, dtype={cfg.get('dtype', 'bfloat16')}",
        f"- chat template: single user turn, add_generation_prompt=True, "
        f"kwargs={gen.get('chat_template_kwargs') or {}}",
        "",
        "## Demonstration selection",
        f"- fixed aligned IDs: {ids}",
        f"- selection: {cfg['prompting']['selection']}",
        "- English evaluation shows the English texts; assigned-language evaluation "
        "shows the assigned-language texts of the same IDs.",
        "",
    ]
    for eval_lang, examples in demos.items():
        for condition in CONDITIONS:
            message = build_prompt(example_text[eval_lang], condition, examples)
            templated, _ = to_model_input(tokenizer, message, gen.get("chat_template_kwargs"))
            lines += [
                f"## {condition} / {eval_lang}",
                "",
                "Message (user turn):",
                "```",
                message,
                "```",
                "",
                "Exact model input after the chat template:",
                "```",
                templated,
                "```",
                "",
            ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prompting.yaml")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument(
        "--sensitivity", action="store_true", help="also run the wording variants (dev only)"
    )
    ap.add_argument(
        "--select-ids",
        action="store_true",
        help="print the five aligned demonstration IDs and exit",
    )
    ap.add_argument("--models", nargs="*", help="only run these roles, e.g. primary")
    ap.add_argument("--limit", type=int, help="first N examples only (smoke test)")
    args = ap.parse_args()

    cfg = yaml.safe_load((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    lang = cfg["language"]
    id_key = cfg["prompting"]["id_key"]
    gen = {**DEFAULT_GENERATION, **cfg.get("generation", {})}

    train_en = load_jsonl(data_path(cfg, "train_en", lang))
    train_tg = load_jsonl(data_path(cfg, "train_target", lang))

    if args.select_ids:
        ids = select_covering_ids(train_en, train_tg, id_key, **cfg["prompting"]["selection"])
        en = {r[id_key]: r for r in train_en}
        tg = {r[id_key]: r for r in train_tg}
        for i in ids:
            print(f"{i}  {gold_labels(en[i]['output'])}")
            print(f"   EN: {en[i]['input']}")
            print(f"   {lang.upper()}: {tg[i]['input']}")
        print("\nPaste into configs/prompting.yaml -> prompting.five_shot_ids:")
        print(json.dumps(ids))
        return

    ids = cfg["prompting"].get("five_shot_ids") or []
    if len(ids) != 5:
        sys.exit("Set exactly five prompting.five_shot_ids first (run --select-ids).")
    if args.split == "test" and args.sensitivity:
        sys.exit("Wording sensitivity is a dev-set analysis only.")

    demos = {
        "en": build_examples(ids, train_en, id_key),
        lang: build_examples(ids, train_tg, id_key),
    }

    eval_sets = {}
    for eval_lang in ["en", lang]:
        rows = load_jsonl(data_path(cfg, args.split, lang, eval_lang))
        eval_sets[eval_lang] = rows[: args.limit] if args.limit else rows

    variants = list(INSTRUCTION_VARIANTS) if args.sensitivity else [DEFAULT_VARIANT]
    pred_dir = PROJECT_ROOT / "outputs" / "predictions" / "prompting" / args.split
    metrics_dir = PROJECT_ROOT / "outputs" / "metrics" / "prompting" / args.split
    summary = []

    for model_cfg in cfg["models"]:
        if args.models and model_cfg["role"] not in args.models:
            continue
        model, tokenizer = load_model(model_cfg["name"], cfg.get("dtype", "bfloat16"))
        model_gen = {**gen, **model_cfg.get("generation", {})}
        tag = model_cfg["name"].split("/")[-1]

        write_prompt_card(
            metrics_dir / f"prompt_card_{tag}.md",
            cfg,
            model_gen,
            ids,
            demos,
            model_cfg,
            tokenizer,
            {k: v[0]["input"] for k, v in eval_sets.items()},
        )

        for eval_lang, rows in eval_sets.items():
            gold = [gold_labels(r["output"]) for r in rows]
            for condition in CONDITIONS:
                for variant in variants:
                    examples = demos[eval_lang] if condition == "five_shot" else None
                    messages = [
                        build_prompt(r["input"], condition, examples, variant) for r in rows
                    ]
                    wrapped = [
                        to_model_input(tokenizer, m, model_gen.get("chat_template_kwargs"))
                        for m in messages
                    ]
                    raw = generate(
                        model,
                        tokenizer,
                        [w[0] for w in wrapped],
                        wrapped[0][1],
                        model_gen["max_new_tokens"],
                        model_gen["batch_size"],
                    )
                    parsed = [parse_prediction(o) for o in raw]
                    pred = [list(p.labels) for p in parsed]
                    metrics = compute_metrics(gold, pred, parsed)
                    metrics["confusion"] = confusion_matrices(gold, pred)

                    run = f"{tag}__{eval_lang}__{condition}__{variant}"
                    write_jsonl(
                        pred_dir / f"{run}.jsonl",
                        [
                            {
                                id_key: r.get(id_key),
                                "text": r["input"],
                                "gold": g,
                                "pred": p,
                                "raw_output": o,
                                "malformed": pr.malformed,
                            }
                            for r, g, p, o, pr in zip(rows, gold, pred, raw, parsed)
                        ],
                    )
                    metrics_dir.mkdir(parents=True, exist_ok=True)
                    (metrics_dir / f"{run}.json").write_text(
                        json.dumps(
                            {
                                "run": run,
                                "fingerprint": prompt_fingerprint(),
                                "generation": model_gen,
                                "five_shot_ids": ids,
                                **metrics,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

                    row = {
                        "model": tag,
                        "role": model_cfg["role"],
                        "lang": eval_lang,
                        "condition": condition,
                        "variant": variant,
                    }
                    row.update({k: v for k, v in metrics.items() if not isinstance(v, dict)})
                    summary.append(row)
                    print(
                        f"{run}: macro_f1={metrics['macro_f1']:.3f} "
                        f"micro_f1={metrics['micro_f1']:.3f} "
                        f"malformed={metrics['malformed_rate']:.3f}"
                    )
        unload(model)

    df = pd.DataFrame(summary)
    name = "summary_sensitivity.csv" if args.sensitivity else "summary.csv"
    df.to_csv(metrics_dir / name, index=False)
    print(
        "\n",
        df.pivot_table(
            index=["model", "lang"], columns=["condition", "variant"], values="macro_f1"
        ).round(3),
    )


if __name__ == "__main__":
    main()
