from __future__ import annotations

import gc
import hashlib
import json
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_project.constants import EMOTIONS
from llm_project.metrics import (
    compute_metrics,
    confusion_matrices,
    gold_labels,
    parse_prediction,
)
from llm_project.utils import load_jsonl

LABELS = list(EMOTIONS)
LABEL_LINE = ", ".join(LABELS)
CONDITIONS = ("zero_shot", "five_shot")
DEFAULT_VARIANT = "default"
DEFAULT_GENERATION = {"max_new_tokens": 64, "batch_size": 8, "chat_template_kwargs": {}}

# these are the original four wording conditions, kept unchanged for reproducibility
INSTRUCTION_VARIANTS: dict[str, str] = {
    "default": (
        "Classify the emotions expressed in the text.\n\n"
        f"The permitted emotion labels are: {LABEL_LINE}.\n\n"
        "A text can express one emotion or several emotions, so return every "
        "permitted label that applies.\n\n"
        "Answer with a JSON list that contains only permitted label names, written "
        'in English exactly as listed (for example ["joy"] or ["anger", "disgust"]). '
        "Do not write anything else."
    ),
    "paraphrase": (
        "Read the text and decide which emotions it conveys.\n\n"
        f"Use only these eight English labels: {LABEL_LINE}.\n\n"
        "More than one label may fit; include all labels that fit, or a single "
        "label if only one fits.\n\n"
        'Respond with nothing but a JSON list of label names (for example ["joy"] '
        'or ["anger", "disgust"]).'
    ),
    "no_format_example": (
        "Classify the emotions expressed in the text.\n\n"
        f"The permitted emotion labels are: {LABEL_LINE}.\n\n"
        "A text can express one emotion or several emotions, so return every "
        "permitted label that applies.\n\n"
        "Answer with a JSON list that contains only permitted label names, written "
        "in English exactly as listed. Do not write anything else."
    ),
    "terse": (
        f"Emotion labels: {LABEL_LINE}.\n"
        "One or several labels may apply.\n"
        "Output only a JSON list of the applicable English label names."
    ),
}


# prompt construction


def format_labels(labels: Sequence[str]) -> str:
    """Return labels in the fixed emotion order without duplicates"""
    wanted = set(labels)
    return json.dumps([label for label in LABELS if label in wanted])


def build_zero_shot_prompt(text: str, variant: str = DEFAULT_VARIANT) -> str:
    return f"{INSTRUCTION_VARIANTS[variant]}\n\nText: {text}\nLabels:"


def build_five_shot_prompt(
    text: str,
    examples: Sequence[dict[str, Any]],
    variant: str = DEFAULT_VARIANT,
) -> str:
    lines = [INSTRUCTION_VARIANTS[variant], "", "Examples:", ""]

    for example in examples:
        lines.extend(
            [
                f"Text: {example['text']}",
                f"Labels: {format_labels(example['labels'])}",
                "",
            ]
        )

    lines.extend(["Now classify this text.", "", f"Text: {text}", "Labels:"])
    return "\n".join(lines)


def build_prompt(
    text: str,
    condition: str,
    examples: Sequence[dict[str, Any]] | None = None,
    variant: str = DEFAULT_VARIANT,
) -> str:
    if condition == "zero_shot":
        return build_zero_shot_prompt(text, variant)
    if condition == "five_shot":
        if not examples:
            raise ValueError("five_shot needs examples")
        return build_five_shot_prompt(text, examples, variant)
    raise ValueError(f"Unknown condition: {condition}")


def prompt_fingerprint() -> str:
    """Identify the exact prompt templates used in an experiment"""
    payload = json.dumps(INSTRUCTION_VARIANTS, sort_keys=True)
    payload += build_five_shot_prompt("X", [{"text": "Y", "labels": ["joy"]}])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# demonstration selection


def select_covering_ids(
    train_en: Sequence[dict[str, Any]],
    train_target: Sequence[dict[str, Any]],
    id_key: str,
    k: int = 5,
    seed: int = 42,
    min_words: int = 5,
    max_words: int = 20,
    max_labels: int = 3,
    min_single: int = 1,
) -> list[Any]:
    """Select fixed aligned examples that cover all eight emotion labels"""
    target_by_id = {row[id_key]: row for row in train_target}
    candidates: list[tuple[Any, frozenset[str]]] = []

    for row in train_en:
        identifier = row[id_key]
        if identifier not in target_by_id:
            continue

        labels = gold_labels(row["output"])
        if not 1 <= len(labels) <= max_labels:
            continue
        if labels != gold_labels(target_by_id[identifier]["output"]):
            continue
        if min_words <= len(row["input"].split()) <= max_words:
            candidates.append((identifier, frozenset(labels)))

    candidates.sort(key=lambda item: str(item[0]))
    rng = random.Random(seed)
    selected: list[tuple[Any, frozenset[str]]] = []
    uncovered = set(LABELS)
    singles_needed = min_single

    for slot in range(k):
        slots_left = k - slot
        already_used = {identifier for identifier, _ in selected}
        pool = [item for item in candidates if item[0] not in already_used]

        # reserve the final slots for single-label examples when requested
        if slots_left <= singles_needed:
            pool = [item for item in pool if len(item[1]) == 1]
            min_new = -(-len(uncovered) // slots_left)
        else:
            multi_slots = slots_left - singles_needed
            min_new = -(-max(0, len(uncovered) - singles_needed) // multi_slots)

        pool = [
            item
            for item in pool
            if len(item[1] & uncovered) >= min_new and (item[1] & uncovered or not uncovered)
        ]
        if not pool:
            raise ValueError("Cannot cover all emotions with these demonstration filters")

        chosen = rng.choice(pool)
        selected.append(chosen)
        uncovered -= chosen[1]
        if len(chosen[1]) == 1:
            singles_needed = max(0, singles_needed - 1)

    if uncovered:
        raise ValueError(f"Emotions left uncovered: {sorted(uncovered)}")
    return [identifier for identifier, _ in selected]


def build_examples(
    ids: Sequence[Any], rows: Sequence[dict[str, Any]], id_key: str
) -> list[dict[str, Any]]:
    """Look up the same demonstration IDs in one language"""
    rows_by_id = {row[id_key]: row for row in rows}
    missing = [identifier for identifier in ids if identifier not in rows_by_id]
    if missing:
        raise KeyError(f"Five-shot IDs missing from the training set: {missing}")

    return [
        {
            "id": identifier,
            "text": rows_by_id[identifier]["input"],
            "labels": gold_labels(rows_by_id[identifier]["output"]),
        }
        for identifier in ids
    ]


# model generation


def to_model_input(
    tokenizer: Any, message: str, chat_template_kwargs: dict | None = None
) -> tuple[str, bool]:
    """Apply the tokenizer chat template when the model provides one"""
    if getattr(tokenizer, "chat_template", None):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            tokenize=False,
            add_generation_prompt=True,
            **(chat_template_kwargs or {}),
        )
        return formatted, True
    return message, False


def generate(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    used_chat_template: bool,
    max_new_tokens: int = 64,
    batch_size: int = 8,
) -> list[str]:
    """Generate one response per prompt using greedy decoding"""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    outputs: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = list(prompts[start : start + batch_size])
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                add_special_tokens=not used_chat_template,
            ).to(model.device)

            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
            input_length = inputs["input_ids"].shape[1]
            outputs.extend(
                tokenizer.batch_decode(generated[:, input_length:], skip_special_tokens=True)
            )
    return outputs


def predict_with_prompt(
    model: Any,
    tokenizer: Any,
    text: str,
    prompt_type: str = "zero_shot",
    examples: Sequence[dict[str, Any]] | None = None,
    config: dict | None = None,
    variant: str = DEFAULT_VARIANT,
) -> str:
    """Convenience function for predicting one sentence in a notebook"""
    generation = {**DEFAULT_GENERATION, **((config or {}).get("generation", {}))}
    message = build_prompt(text, prompt_type, examples, variant)
    formatted, used_template = to_model_input(
        tokenizer, message, generation.get("chat_template_kwargs")
    )
    return generate(
        model,
        tokenizer,
        [formatted],
        used_template,
        generation["max_new_tokens"],
        batch_size=1,
    )[0]


# dataset and model loading


def data_path(
    project_root: Path,
    config: dict,
    key: str,
    target_lang: str,
    input_lang: str | None = None,
) -> Path:
    data = config["data"]
    folder = project_root / data["dir"].format(lang=target_lang)
    filename = data[key]
    if isinstance(filename, dict):
        filename = filename["en" if input_lang == "en" else "target"]
    return folder / filename.format(lang=target_lang)


def read_rows(path: Path) -> list[dict[str, Any]]:
    # use the existing jsonl loader instead of implementing another one
    rows = cast(list[dict[str, Any]], load_jsonl(path).to_list())
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def load_model(model_name: str, dtype: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, dtype),
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def load_evaluation_data(
    project_root: Path,
    config: dict,
    target_lang: str,
    split: str,
    limit: int | None,
) -> dict[str, list[dict[str, Any]]]:
    datasets = {}
    for input_lang in ("en", target_lang):
        rows = read_rows(data_path(project_root, config, split, target_lang, input_lang))
        datasets[input_lang] = rows[:limit] if limit is not None else rows
    return datasets


def load_selected_variants(project_root: Path, split: str, target_lang: str) -> dict[str, str]:
    if split != "test":
        return {}

    path = project_root / "outputs/report" / f"selected_prompt_variants_{target_lang}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; select prompts on DEV before running TEST")
    return json.loads(path.read_text(encoding="utf-8"))


# result saving


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_prompt_card(
    path: Path,
    config: dict,
    generation: dict,
    ids: Sequence[Any],
    demonstrations: dict[str, list[dict[str, Any]]],
    model_config: dict,
    tokenizer: Any,
    datasets: dict[str, list[dict[str, Any]]],
) -> None:
    """Save an example of each complete prompt and model chat template"""
    lines = [
        f"# Prompt card: {model_config['name']} ({model_config['role']})",
        "",
        f"Prompt fingerprint: `{prompt_fingerprint()}`",
        "",
        "## Decoding",
        f"- greedy decoding (do_sample=False), max_new_tokens={generation['max_new_tokens']}",
        f"- batch_size={generation['batch_size']}, left padding, dtype={config.get('dtype', 'bfloat16')}",
        f"- chat template kwargs={generation.get('chat_template_kwargs') or {}}",
        "",
        "## Demonstration selection",
        f"- fixed aligned IDs: {list(ids)}",
        f"- selection: {config['prompting']['selection']}",
        "- English and Swedish prompts use the same aligned demonstration IDs",
        "",
    ]

    for input_lang, examples in demonstrations.items():
        example_text = datasets[input_lang][0]["input"]
        for condition in CONDITIONS:
            for variant in INSTRUCTION_VARIANTS:
                prompt = build_prompt(example_text, condition, examples, variant)
                formatted, _ = to_model_input(
                    tokenizer, prompt, generation.get("chat_template_kwargs")
                )
                lines.extend(
                    [
                        f"## {condition} / {input_lang} / {variant}",
                        "",
                        "Message (user turn):",
                        "```",
                        prompt,
                        "```",
                        "",
                        "Exact model input after the chat template:",
                        "```",
                        formatted,
                        "```",
                        "",
                    ]
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_name(
    model_name: str, target_lang: str, input_lang: str, condition: str, variant: str
) -> str:
    model_tag = model_name.split("/")[-1]
    return f"{model_tag}__{target_lang}__{input_lang}__{condition}__{variant}"


def run_experiment(
    *,
    model: Any,
    tokenizer: Any,
    model_config: dict,
    generation: dict,
    rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    target_lang: str,
    input_lang: str,
    condition: str,
    variant: str,
    split: str,
    id_key: str,
    ids: Sequence[Any],
    seed: int,
    prediction_dir: Path,
    metrics_dir: Path,
) -> dict[str, Any]:
    """Run and save one model / language / shot / wording combination"""
    demo_examples = examples if condition == "five_shot" else None
    prompts = [build_prompt(row["input"], condition, demo_examples, variant) for row in rows]
    model_inputs = [
        to_model_input(tokenizer, prompt, generation.get("chat_template_kwargs"))
        for prompt in prompts
    ]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    raw_outputs = generate(
        model,
        tokenizer,
        [item[0] for item in model_inputs],
        model_inputs[0][1],
        generation["max_new_tokens"],
        generation["batch_size"],
    )
    elapsed = time.perf_counter() - start

    if len(raw_outputs) != len(rows):
        raise ValueError("Number of model outputs does not match the evaluation dataset")

    # use the same parser and scoring functions as the peft evaluation
    parsed = [parse_prediction(raw) for raw in raw_outputs]
    gold = [gold_labels(row["output"]) for row in rows]
    predicted = [list(result.labels) for result in parsed]
    scores = compute_metrics(gold, predicted, parsed)
    scores["confusion"] = confusion_matrices(gold, predicted)

    name = run_name(model_config["name"], target_lang, input_lang, condition, variant)
    prediction_rows = [
        {
            id_key: row.get(id_key),
            "text": row["input"],
            "gold": labels,
            "pred": prediction,
            "raw_output": raw,
            "malformed": result.malformed,
        }
        for row, labels, prediction, raw, result in zip(
            rows, gold, predicted, raw_outputs, parsed, strict=True
        )
    ]
    write_jsonl(prediction_dir / f"{name}.jsonl", prediction_rows)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    peak_memory = (
        round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None
    )
    metadata = {
        "run": name,
        "split": split,
        "target_language": target_lang,
        "evaluation_language": input_lang,
        "model": model_config["name"],
        "method": condition,
        "variant": variant,
        "inference_seconds": round(elapsed, 3),
        "random_seed": seed,
        "gpu": gpu_name,
        "peak_gpu_memory_gb": peak_memory,
        "fingerprint": prompt_fingerprint(),
        "generation": generation,
        "five_shot_ids": list(ids),
        **scores,
    }
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{name}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        f"{name}: micro_f1={scores['micro_f1']:.3f} "
        f"macro_f1={scores['macro_f1']:.3f} "
        f"malformed={scores['malformed_rate']:.3f}"
    )

    return {
        "model": model_config["name"].split("/")[-1],
        "role": model_config["role"],
        "lang": input_lang,
        "condition": condition,
        "variant": variant,
        **{key: value for key, value in scores.items() if not isinstance(value, dict)},
    }


# full workflow


def load_shared_training_data(project_root: Path, config: dict) -> dict[str, list[dict[str, Any]]]:
    """Load aligned English and Swedish training examples"""
    return {
        "en": read_rows(data_path(project_root, config, "train_en", "sv")),
        "sv": read_rows(data_path(project_root, config, "train_target", "sv")),
    }


def find_shared_ids(datasets: dict[str, list[dict[str, Any]]], id_key: str) -> list[Any]:
    """Find English-Swedish alignment IDs with matching emotion labels"""
    english = {row[id_key]: row for row in datasets["en"]}
    swedish = {row[id_key]: row for row in datasets["sv"]}

    # reject repeated alignment IDs instead of silently overwriting examples
    if len(english) != len(datasets["en"]) or len(swedish) != len(datasets["sv"]):
        raise ValueError("Duplicate alignment IDs in the English or Swedish training data")

    shared = set(english) & set(swedish)
    return [
        identifier
        for identifier in sorted(shared, key=str)
        if gold_labels(english[identifier]["output"]) == gold_labels(swedish[identifier]["output"])
    ]


def select_shared_ids(config: dict, datasets: dict[str, list[dict[str, Any]]]) -> list[Any]:
    """Select five aligned English-Swedish demonstrations covering all emotions"""
    id_key = config["prompting"]["id_key"]
    shared = set(find_shared_ids(datasets, id_key))
    if not shared:
        raise ValueError(
            "No aligned English-Swedish TRAIN examples with matching labels were found"
        )

    english = [row for row in datasets["en"] if row[id_key] in shared]
    swedish = [row for row in datasets["sv"] if row[id_key] in shared]
    try:
        return select_covering_ids(english, swedish, id_key, **config["prompting"]["selection"])
    except ValueError as exc:
        raise ValueError(
            f"Found {len(shared)} aligned English-Swedish TRAIN examples, "
            "but could not select five that cover all eight emotions "
            "under the configured selection constraints"
        ) from exc


def show_demonstration_ids(config: dict, datasets: dict[str, list[dict[str, Any]]]) -> None:
    """Print the five aligned English-Swedish demonstration IDs"""
    id_key = config["prompting"]["id_key"]
    ids = select_shared_ids(config, datasets)
    by_language = {
        language: {row[id_key]: row for row in rows} for language, rows in datasets.items()
    }

    for identifier in ids:
        print(f"{identifier}  {gold_labels(by_language['en'][identifier]['output'])}")
        for language in ("en", "sv"):
            print(f"   {language.upper()}: {by_language[language][identifier]['input']}")

    print("\nCopy into configs/prompting.yaml -> prompting.five_shot_ids:")
    print(json.dumps(ids, ensure_ascii=False))


def get_five_shot_ids(config: dict[str, Any]) -> list[Any]:
    """Read one shared list of five demonstration IDs"""
    ids = config["prompting"].get("five_shot_ids")
    if not isinstance(ids, list) or len(ids) != 5 or len(set(ids)) != 5:
        raise ValueError(
            "Set prompting.five_shot_ids to one list of five distinct IDs "
            "(run scripts/run_prompting.py --select-ids first)"
        )
    return ids


def validate_shared_ids(
    ids: Sequence[Any],
    datasets: dict[str, list[dict[str, Any]]],
    id_key: str,
) -> None:
    """Verify every demonstration is aligned across English and Swedish"""
    valid = set(find_shared_ids(datasets, id_key))
    invalid = [identifier for identifier in ids if identifier not in valid]
    if invalid:
        raise ValueError(
            f"Five-shot IDs are missing or have conflicting labels in "
            f"English-Swedish TRAIN pairs: {invalid}. "
            "Run --select-ids to choose valid shared IDs."
        )


def run_prompting_experiments(
    project_root: Path,
    config_path: str = "configs/prompting.yaml",
    language: str | None = None,
    split: str = "dev",
    sensitivity: bool = False,
    select_ids: bool = False,
    model_roles: Sequence[str] | None = None,
    conditions: Sequence[str] | None = None,
    limit: int | None = None,
    skip_existing: bool = False,
) -> None:
    """Run English and Swedish prompting experiments for the Swedish dataset"""
    if split == "test" and (sensitivity or limit is not None):
        raise ValueError("TEST does not allow sensitivity tuning or a limited evaluation")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = project_root / config_file
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))

    target_lang = language or config.get("language", "sv")
    if target_lang != "sv":
        raise ValueError("This prompting pipeline supports Swedish (sv) only")

    # selection requires aligned English-Swedish TRAIN pairs only
    if select_ids:
        shared_data = load_shared_training_data(project_root, config)
        show_demonstration_ids(config, shared_data)
        return

    conditions_to_run = conditions if conditions is not None else CONDITIONS
    if set(conditions_to_run) - set(CONDITIONS):
        raise ValueError(f"Unknown prompting conditions: {conditions_to_run}")

    available_roles = {model["role"] for model in config["models"]}
    unknown_roles = set(model_roles or ()) - available_roles
    if unknown_roles:
        raise ValueError(
            f"Unknown model roles: {sorted(unknown_roles)}; available: {sorted(available_roles)}"
        )

    prompt_config = config["prompting"]
    id_key = prompt_config["id_key"]
    shared_data = load_shared_training_data(project_root, config)
    ids = get_five_shot_ids(config)
    validate_shared_ids(ids, shared_data, id_key)
    demos = {
        "en": build_examples(ids, shared_data["en"], id_key),
        target_lang: build_examples(ids, shared_data[target_lang], id_key),
    }
    datasets = load_evaluation_data(project_root, config, target_lang, split, limit)
    selected = load_selected_variants(project_root, split, target_lang)
    variants = list(INSTRUCTION_VARIANTS) if sensitivity else [DEFAULT_VARIANT]

    folder = f"{split}_smoke" if limit is not None else split
    prediction_dir = project_root / "outputs/predictions/prompting" / folder
    metrics_dir = project_root / "outputs/metrics/prompting" / folder
    metrics_dir.mkdir(parents=True, exist_ok=True)
    default_generation = {**DEFAULT_GENERATION, **config.get("generation", {})}
    summaries = []

    for model_config in config["models"]:
        if model_roles and model_config["role"] not in model_roles:
            continue

        model, tokenizer = load_model(model_config["name"], config.get("dtype", "bfloat16"))
        generation = {**default_generation, **model_config.get("generation", {})}
        tag = model_config["name"].split("/")[-1]

        try:
            write_prompt_card(
                metrics_dir / f"prompt_card_{tag}_{target_lang}.md",
                config,
                generation,
                ids,
                demos,
                model_config,
                tokenizer,
                datasets,
            )
            for input_lang, rows in datasets.items():
                for condition in conditions_to_run:
                    key = f"{tag}__{target_lang}__{input_lang}__{condition}"
                    run_variants = [selected[key]] if split == "test" else variants

                    for variant in run_variants:
                        name = f"{key}__{variant}"
                        metric_file = metrics_dir / f"{name}.json"
                        prediction_file = prediction_dir / f"{name}.jsonl"
                        if skip_existing and metric_file.exists() and prediction_file.exists():
                            if condition == "five_shot":
                                old_metrics = json.loads(metric_file.read_text(encoding="utf-8"))
                                if old_metrics.get("five_shot_ids") != list(ids):
                                    raise ValueError(
                                        f"Existing five-shot result {name} used different "
                                        "demonstrations. Rerun this experiment without "
                                        "--skip-existing before comparing results."
                                    )
                            print(f"Skipping completed experiment: {name}")
                            continue

                        summary = run_experiment(
                            model=model,
                            tokenizer=tokenizer,
                            model_config=model_config,
                            generation=generation,
                            rows=rows,
                            examples=demos[input_lang],
                            target_lang=target_lang,
                            input_lang=input_lang,
                            condition=condition,
                            variant=variant,
                            split=split,
                            id_key=id_key,
                            ids=ids,
                            seed=prompt_config["selection"]["seed"],
                            prediction_dir=prediction_dir,
                            metrics_dir=metrics_dir,
                        )
                        summaries.append(summary)
        finally:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if summaries:
        roles = "_".join(model_roles) if model_roles else "all"
        filename = "summary_sensitivity.csv" if sensitivity else "summary.csv"
        frame = pd.DataFrame(summaries)
        frame.to_csv(metrics_dir / f"{target_lang}_{roles}_{filename}", index=False)
        table = frame.pivot_table(
            index=["model", "lang"],
            columns=["condition", "variant"],
            values="macro_f1",
        )
        print("\n", table.round(3))
