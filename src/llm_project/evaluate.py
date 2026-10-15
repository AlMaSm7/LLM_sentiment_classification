import ast
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

MODELS = {"qwen": "Qwen/Qwen2.5-7B-Instruct", "phi": "microsoft/Phi-3.5-mini-instruct"}
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
SEED = 42
MAX_NEW_TOKENS = 40
MAX_INPUT_TOKENS = 2048
USE_4BIT = True

PROMPT_V1_SYSTEM = (
    "You are an expert emotion classifier for movie subtitles. "
    "Your task is to identify all emotions expressed in a single subtitle. "
    f"Choose exclusively from these eight emotion labels: {', '.join(EMOTIONS)}.\n\n"
    "Emotion definitions: "
    "anger = frustration, hostility, or irritation; "
    "anticipation = expectation, curiosity, or looking forward to something; "
    "disgust = aversion, contempt, or strong disapproval; "
    "fear = worry, anxiety, concern, or perceived danger; "
    "joy = happiness, pleasure, excitement, or satisfaction; "
    "sadness = grief, disappointment, loneliness, or unhappiness; "
    "surprise = astonishment, unexpectedness, or sudden realization; "
    "trust = confidence, reassurance, acceptance, or belief in someone.\n\n"
    "Classification rules: "
    "Select every emotion supported by the subtitle. "
    "Multiple emotions may occur simultaneously. "
    "Consider both directly stated emotions and emotions implied by the wording. "
    "Do not invent emotions that are not supported by the text.\n\n"
    "Strict output formatting rules: "
    "Return exactly one valid JSON array. "
    "Every emotion label must be enclosed in double quotation marks. "
    'Correct: ["fear"], ["anger", "surprise"], or []. '
    "Incorrect: [fear], ['fear'], fear, or fear, surprise. "
    "Never omit quotation marks around emotion labels. "
    "Never include explanations, Markdown, or text outside the JSON array."
)
PROMPT_V1_USER = "Subtitle: {text}\nEmotions (JSON):"

PROMPT_V2_SYSTEM = (
    "You are analyzing emotional meaning in dialogue from movies and television. "
    "Your task is to identify the emotions conveyed by a single subtitle, "
    "even when those emotions are not explicitly named. "
    f"Use only these eight emotion labels: {', '.join(EMOTIONS)}.\n\n"
    "Interpretation guidelines: "
    "Movie dialogue often communicates emotions indirectly. "
    "Consider the speaker's likely emotional intention, tone, and attitude. "
    "Questions may express curiosity or anticipation. "
    "Warnings and expressions of concern may communicate fear. "
    "Compliments, encouragement, and reassurance may communicate joy or trust. "
    "Criticism, threats, and confrontation may communicate anger or disgust. "
    "Unexpected events or discoveries may communicate surprise. "
    "Expressions of loss, regret, or disappointment may communicate sadness.\n\n"
    "A subtitle can express several emotions at once. "
    "Identify all emotions reasonably supported by the dialogue. "
    "Do not require explicit emotional words to assign a label. "
    "However, do not assume details about the surrounding scene that are not provided.\n\n"
    "Strict output formatting rules: "
    "Return exactly one valid JSON array. "
    "Every emotion label must be enclosed in double quotation marks. "
    'Correct: ["fear"], ["anger", "surprise"], or []. '
    "Incorrect: [fear], ['fear'], fear, or fear, surprise. "
    "Never omit quotation marks around emotion labels. "
    "Never include explanations, Markdown, or text outside the JSON array."
)
PROMPT_V2_USER = "Movie dialogue: {text}\nEmotion labels (JSON):"

PROMPTS = {1: (PROMPT_V1_SYSTEM, PROMPT_V1_USER), 2: (PROMPT_V2_SYSTEM, PROMPT_V2_USER)}


def parse_answer(answer):
    text = answer.strip().strip("` ").strip()
    text = re.sub(r"^(emotions?|labels?|answer)\s*:\s*", "", text, flags=re.I).strip()

    if text.lower() in {"none", "neutral", "[]"}:
        return []

    # parse json-like lists, including unquoted labels
    if text.startswith("[") and text.endswith("]"):
        try:
            labels = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            labels = [label.strip() for label in text[1:-1].split(",")]

        if not isinstance(labels, list):
            return None
    else:
        labels = re.split(r"[,;\n]+", text)

    labels = [str(label).strip().strip("\"' ").lower() for label in labels]

    if not labels or any(label not in EMOTIONS for label in labels):
        return None

    return sorted(set(labels))


def read_rows(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run prepare.py first")
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"No evaluation data in {path}")
    return rows


def make_messages(text, system, user_template, examples, language):
    messages = [{"role": "system", "content": system}]
    for row in examples:
        messages.append({"role": "user", "content": user_template.format(text=row[language])})
        messages.append({"role": "assistant", "content": json.dumps(row["labels"])})
    messages.append({"role": "user", "content": user_template.format(text=text)})
    return messages


def choose_prompt_version(args):
    if args.method == "peft":
        if args.shots != 0 or args.version not in ("auto", "1") or args.model != "qwen":
            raise ValueError(
                "PEFT evaluates only Qwen, no few-shot demonstrations and training prompt v1"
            )
        return 1
    if args.split == "test":
        if args.version != "auto":
            raise ValueError("TEST requires --version auto from saved DEV-only prompt selection")
        path = Path("outputs/prompt_selection.json")
        if not path.exists():
            raise FileNotFoundError(
                "Run python results.py --split dev --select before any TEST prompting"
            )
        selected = json.loads(path.read_text(encoding="utf-8"))
        key = f"{args.target_language}_{args.model}_s{args.shots}"
        if key not in selected:
            raise KeyError(f"Missing DEV-selected prompt version for {key}")
        return int(selected[key])
    if args.version == "auto":
        raise ValueError("DEV evaluations require an explicit --version 1 or --version 2")
    return int(args.version)


def experiment_name(args):
    target = args.target_language
    input_lang = args.input_language

    if args.method == "peft":
        training = {
            "en": "en",
            "target": target,
            "bilingual": f"en_and_{target}",
        }[args.condition]

        method = f"trained_on_{training}"
    else:
        shots = "five_shot" if args.shots == 5 else "zero_shot"
        method = f"{shots}_v{args.version}"

    return f"{args.model}_{method}_using_{input_lang}_input_for_{target}_target_on_{args.split}"


def evaluate(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Inference requires a CUDA GPU")
    if args.method == "peft" and args.condition not in {"en", "target", "bilingual"}:
        raise ValueError("PEFT requires --condition en, target or bilingual")
    if args.method == "peft" and args.condition == "target" and args.input_language == "en":
        raise ValueError("Target-language-only PEFT is evaluated on the target language")
    if args.method == "prompt" and args.condition != "none":
        raise ValueError("Prompt-only models have no PEFT condition")
    set_seed(SEED)
    version = choose_prompt_version(args)
    folder = Path("data/processed") / args.target_language
    rows = read_rows(folder / f"{args.split}.jsonl")

    shots = []
    if args.method == "prompt" and args.shots == 5:
        shots_file = folder / "five_shot.json"
        shots = json.loads(shots_file.read_text(encoding="utf-8"))
        train_ids = {row["id"] for row in read_rows(folder / "train.jsonl")}
        current_ids = {row["id"] for row in rows}
        if len(shots) != 5 or len({row["id"] for row in shots}) != 5:
            raise ValueError("Exactly five unique aligned training examples are required")
        if any(row["id"] not in train_ids or row["id"] in current_ids for row in shots):
            raise ValueError("Five-shot demonstration leaked outside training data")

    system, user_template = PROMPTS[1 if args.method == "peft" else version]
    model_id = MODELS[args.model]
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
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
        quantization_config=quantization,
        trust_remote_code=False,
    )
    if args.method == "peft":
        adapter = Path("outputs/models") / args.target_language / args.condition / "adapter"
        if not adapter.exists():
            raise FileNotFoundError(f"Missing trained adapter at {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    name = experiment_name(args)
    prediction_path = Path("outputs/predictions") / f"{name}.jsonl"
    metric_path = Path("outputs/metrics") / f"{name}.json"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    if (prediction_path.exists() or metric_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Results for {name} already exist; use --overwrite to rerun deliberately"
        )

    output = []
    temp_path = prediction_path.with_name(f"{prediction_path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as file:
            for number, row in enumerate(rows, 1):
                msgs = make_messages(
                    row[args.input_language],
                    system,
                    user_template,
                    shots,
                    args.input_language,
                )
                # tokenize prompt with attention mask
                inputs = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)

                # check input length
                if inputs["input_ids"].shape[1] > MAX_INPUT_TOKENS:
                    raise ValueError("Prompt exceeds maximum input length")

                # generate prediction
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                # decode generated answer
                input_length = inputs["input_ids"].shape[1]

                raw = tokenizer.decode(
                    generated[0, input_length:],
                    skip_special_tokens=True,
                )
                # parse model prediction
                predicted = parse_answer(raw)
                malformed = predicted is None

                if malformed:
                    predicted = []
                item = {
                    "id": row["id"],
                    "text": row[args.input_language],
                    "gold": row["labels"],
                    "prediction": predicted,
                    "malformed": malformed,
                    "raw": raw,
                }
                output.append(item)
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
                if number % 50 == 0:
                    print(f"{name}: {number}/{len(rows)}", flush=True)
        temp_path.replace(prediction_path)
    finally:
        temp_path.unlink(missing_ok=True)

    gold = np.array([[int(label in row["gold"]) for label in EMOTIONS] for row in output])
    pred = np.array([[int(label in row["prediction"]) for label in EMOTIONS] for row in output])
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        gold, pred, average="micro", zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        gold, pred, average="macro", zero_division=0
    )
    p, r, f, support = precision_recall_fscore_support(gold, pred, average=None, zero_division=0)
    metrics = {
        "run": name,
        "lang": args.target_language,
        "split": args.split,
        "method": args.method,
        "model": args.model,
        "condition": args.condition,
        "input_language": args.input_language,
        "shots": args.shots,
        "prompt_version": version,
        "n": len(output),
        "seed": SEED,
        "micro_f1": float(micro_f1),
        "sample_jaccard": float(jaccard_score(gold, pred, average="samples", zero_division=0)),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "hamming_loss": float(hamming_loss(gold, pred)),
        "exact_match": float(accuracy_score(gold, pred)),
        "malformed_rate": sum(row["malformed"] for row in output) / len(output),
        "per_emotion": {
            emotion: {
                "precision": float(p[i]),  # type: ignore
                "recall": float(r[i]),  # type: ignore
                "f1": float(f[i]),  # type: ignore
                "support": int(support[i]),  # type: ignore
            }
            for i, emotion in enumerate(EMOTIONS)
        },
    }
    temp_metrics = metric_path.with_name(f"{metric_path.name}.{os.getpid()}.tmp")
    try:
        temp_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_metrics.replace(metric_path)
    finally:
        temp_metrics.unlink(missing_ok=True)
    print(f"{name}: micro_f1={micro_f1:.4f}, sample_jaccard={metrics['sample_jaccard']:.4f}")
