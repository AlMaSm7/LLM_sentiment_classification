from __future__ import annotations
import hashlib
import json
import random
import re

# from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

LABELS: list[str] = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]
_LABEL_SET = frozenset(LABELS)
_LABEL_LINE = ", ".join(LABELS)

# dictionary of four instructions, default is the actual experimental prompt
INSTRUCTION_VARIANTS: dict[str, str] = {
    "default": (
        "Classify the emotions expressed in the text.\n\n"
        f"The permitted emotion labels are: {_LABEL_LINE}.\n\n"
        "A text can express one emotion or several emotions, so return every "
        "permitted label that applies.\n\n"
        "Answer with a JSON list that contains only permitted label names, written "
        'in English exactly as listed (for example ["joy"] or ["anger", "disgust"]). '
        "Do not write anything else."
    ),
    # Same content and same format example, different wording.
    "paraphrase": (
        "Read the text and decide which emotions it conveys.\n\n"
        f"Use only these eight English labels: {_LABEL_LINE}.\n\n"
        "More than one label may fit; include all labels that fit, or a single "
        "label if only one fits.\n\n"
        'Respond with nothing but a JSON list of label names (for example ["joy"] '
        'or ["anger", "disgust"]).'
    ),
    # Default wording without the format example (tests the example's influence).
    "no_format_example": (
        "Classify the emotions expressed in the text.\n\n"
        f"The permitted emotion labels are: {_LABEL_LINE}.\n\n"
        "A text can express one emotion or several emotions, so return every "
        "permitted label that applies.\n\n"
        "Answer with a JSON list that contains only permitted label names, written "
        "in English exactly as listed. Do not write anything else."
    ),
    # Minimal wording.
    "terse": (
        f"Emotion labels: {_LABEL_LINE}.\n"
        "One or several labels may apply.\n"
        "Output only a JSON list of the applicable English label names."
    ),
}
DEFAULT_VARIANT = "default"


# no duplicates, important for ico learning
def format_labels(labels):
    return json.dumps([label for label in LABELS if label in set(labels)])


def build_zero_shot_prompt(text, variant=DEFAULT_VARIANT):
    return f"{INSTRUCTION_VARIANTS[variant]}\n\nText: {text}\nLabels:"


def build_five_shot_prompt(text, examples, variant=DEFAULT_VARIANT):
    """Same instruction as zero-shot; the only difference is the demonstrations.

    `examples` items need "text" and "labels" (a list of label names).
    """
    lines = [INSTRUCTION_VARIANTS[variant], "", "Examples:", ""]
    for ex in examples:
        lines += [f"Text: {ex['text']}", f"Labels: {format_labels(ex['labels'])}", ""]
    lines += ["Now classify this text.", "", f"Text: {text}", "Labels:"]
    return "\n".join(lines)


# given the condition zero/five --> calls the right builder
def build_prompt(text, condition, examples=None, variant=DEFAULT_VARIANT):
    if condition == "zero_shot":
        return build_zero_shot_prompt(text, variant)
    if condition == "five_shot":
        if not examples:
            raise ValueError("five_shot needs examples")
        return build_five_shot_prompt(text, examples, variant)
    raise ValueError(f"Unknown condition: {condition}")


# turn the instructions into one string, hash it and return the first 16 hex digits.
# this is used to check that the instructions were not changed after evaluation.
def prompt_fingerprint():
    """Hash of all templates. Store it with dev and test results to show the
    prompts were not changed after test evaluation."""
    text = json.dumps(INSTRUCTION_VARIANTS, sort_keys=True)
    text += build_five_shot_prompt("X", [{"text": "Y", "labels": ["joy"]}])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


CODE_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)  # ```json ... ```
FIRST_LIST = re.compile(r"\[[^\[\]]*\]", re.DOTALL)  # the first [...] in the text


@dataclass(frozen=True)
class ParsedOutput:
    labels: tuple  # permitted labels found in answer
    malformed: bool  # TRUE if no valid JSON
    strict_json: bool  # TRUE if it was a json

    @property
    def empty(self):
        return not self.labels


# case + whitespace normalization
def normalise(label):
    return " ".join(label.split()).lower()


def read_json(txt):
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return None


def is_list_of_str(val):
    return isinstance(val, list) and all(isinstance(x, str) for x in val)


# only keep the permitted lavels, drop duplicates, normalise case/whitespace
def parse_prediction(raw):
    text = raw.strip()
    strict = is_list_of_str(read_json(text))
    code_block = CODE_BLOCK.search(text)
    if code_block:
        text = code_block.group(1).strip()
    first_list = FIRST_LIST.search(text)
    items = read_json(first_list.group(0)) if first_list else None
    if not is_list_of_str(items):
        return ParsedOutput(labels=(), malformed=True, strict_json=False)

    found = {normalise(item) for item in items} & _LABEL_SET
    return ParsedOutput(
        labels=tuple(label for label in LABELS if label in found),
        malformed=False,
        strict_json=strict,
    )


def parse_labels(raw):
    return list(parse_prediction(raw).labels)


def ceil_div(a, b):
    return -(-a // b)


def gold_labels(output):
    if not isinstance(output, list):
        parsed = parse_prediction(str(output))
        if parsed.malformed:
            raise ValueError(f"Could not read gold labels: {output!r}")
        return list(parsed.labels)

    labels = {normalise(label) for label in output}
    unknown = labels - _LABEL_SET
    if unknown:
        raise ValueError(f"Gold labels outside the label set: {unknown}")
    return [label for label in LABELS if label in labels]


# here we get the 5 prompt shots, only the ones that are in both train sets, and have the same labels, and have 1-3 labels, and have 5-20 words
def select_covering_ids(
    train_en,
    train_target,
    id_key,
    k=5,
    seed=42,
    min_words=5,
    max_words=20,
    max_labels=3,
    min_single=1,
):
    target_by_id = {r[id_key]: r for r in train_target}
    candidates = []
    for r in train_en:
        rid = r[id_key]
        if rid not in target_by_id:
            continue
        labels = gold_labels(r["output"])
        if not 1 <= len(labels) <= max_labels:
            continue
        if labels != gold_labels(target_by_id[rid]["output"]):
            continue
        if min_words <= len(r["input"].split()) <= max_words:
            candidates.append((rid, frozenset(labels)))
    candidates.sort(key=lambda c: str(c[0]))

    rng = random.Random(seed)
    chosen = []
    uncovered = set(LABELS)
    singles_needed = min_single

    for slot in range(k):
        slots_left = k - slot
        chosen_ids = {row_id for row_id, _ in chosen}
        pool = [c for c in candidates if c[0] not in chosen_ids]

        if slots_left <= singles_needed:
            pool = [c for c in pool if len(c[1]) == 1]
            new_labels_needed = ceil_div(len(uncovered), slots_left)
        else:
            multi_slots = slots_left - singles_needed
            new_labels_needed = ceil_div(max(0, len(uncovered) - singles_needed), multi_slots)

        pool = [
            c
            for c in pool
            if len(c[1] & uncovered) >= new_labels_needed and (c[1] & uncovered or not uncovered)
        ]
        if not pool:
            raise ValueError("Cannot cover all labels with these filters; relax them or raise k.")

        row_id, labels = rng.choice(pool)
        chosen.append((row_id, labels))
        uncovered -= labels
        if len(labels) == 1:
            singles_needed = max(0, singles_needed - 1)

    if uncovered:
        raise ValueError(f"Labels left uncovered: {sorted(uncovered)}")
    return [row_id for row_id, _ in chosen]


# returns the demonstrations as {"id", "text", "labels"} dictionaries.
def build_examples(ids, rows, id_key):
    by_id = {r[id_key]: r for r in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"Five-shot IDs missing from this train set: {missing}")
    return [
        {"id": i, "text": by_id[i]["input"], "labels": gold_labels(by_id[i]["output"])} for i in ids
    ]


DEFAULT_GENERATION = {"max_new_tokens": 64, "batch_size": 8, "chat_temp_kwargs": {}}


def to_model_input(tokenizer, msg, chat_temp_kwargs):
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": msg}],
            tokenize=False,
            add_generation_prompt=True,
            **(chat_temp_kwargs or {}),
        )
        return prompt, True
    return msg, False


def generate(model, tokenizer, prompts, used_chat_template, max_new_tokens=64, batch_size=8):
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    answers = []
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
            new_tokens = generated[:, inputs["input_ids"].shape[1] :]
            answers += tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return answers


def predict_with_prompt(
    model,
    tokenizer,
    text,
    prompt_type="zero_shot",
    examples=None,
    variant=DEFAULT_VARIANT,
    config=None,
):
    settings = {**DEFAULT_GENERATION, **((config or {}).get("generation", {}))}
    msg = build_prompt(text, prompt_type, examples, variant)
    prompt, used = to_model_input(tokenizer, msg, settings.get("chat_temp_kwargs"))
    return generate(model, tokenizer, [prompt], used, settings["max_new_tokens"], 1)[0]
