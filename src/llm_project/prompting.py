"""Phase 2: zero-shot and five-shot prompting for multi-label emotion classification.

Everything that defines the experimental condition lives here, so the report can
point to one file for: the prompt templates, the demonstration-selection
procedure, the output parser and the decoding parameters.

uv run python scripts/run_prompting.py --select-ids
to get the ids with multiple feelings so we cover more in the prompts

"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Sequence
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

#dictionary of four instructions, default is the actual experimental prompt
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

#no duplicates, important for ico learning
def format_labels(labels: Sequence[str]) -> str:
    """Canonical JSON rendering of a label set (fixed label order, no duplicates)."""
    wanted = set(labels)
    return json.dumps([label for label in LABELS if label in wanted])


def build_zero_shot_prompt(text: str, variant: str = DEFAULT_VARIANT) -> str:
    return f"{INSTRUCTION_VARIANTS[variant]}\n\nText: {text}\nLabels:"


def build_five_shot_prompt(
    text: str,
    examples: Sequence[dict[str, Any]],
    variant: str = DEFAULT_VARIANT,
) -> str:
    """Same instruction as zero-shot; the only difference is the demonstrations.

    `examples` items need "text" and "labels" (a list of label names).
    """
    lines = [INSTRUCTION_VARIANTS[variant], "", "Examples:", ""]
    for ex in examples:
        lines += [f"Text: {ex['text']}", f"Labels: {format_labels(ex['labels'])}", ""]
    lines += ["Now classify this text.", "", f"Text: {text}", "Labels:"]
    return "\n".join(lines)

#given the condition zero/five --> calls the right builder
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

#turn the instructions into one string, hash it and return the first 16 hex digits. 
# this is used to check that the instructions were not changed after evaluation.
def prompt_fingerprint() -> str:
    """Hash of all templates. Store it with dev and test results to show the
    prompts were not changed after test evaluation."""
    payload = json.dumps(INSTRUCTION_VARIANTS, sort_keys=True)
    payload += build_five_shot_prompt("X", [{"text": "Y", "labels": ["joy"]}])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 2. Output parsing
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_LIST_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)


@dataclass(frozen=True)
class ParsedOutput:
    labels: tuple[str, ...]
    malformed: bool  # no parseable JSON list of strings in the output
    strict_json: bool  # the whole output (whitespace-stripped) is exactly a JSON list

    @property
    def empty(self) -> bool:
        return not self.labels


def _normalise(item: str) -> str:
    """Case and whitespace normalisation only. No synonym mapping."""
    return " ".join(item.split()).lower()


def parse_prediction(raw: str) -> ParsedOutput:
    """Parse a model output into a set of permitted labels.

    Rules (report these):
      * strip surrounding whitespace and an optional ```json fence;
      * take the first [...] span and json.loads it;
      * if that fails, or it is not a list of strings -> malformed, empty set;
      * normalise case/whitespace of each item, keep only exact permitted labels,
        drop duplicates; anything else (e.g. "happiness") is ignored, not mapped;
      * a well-formed list with no permitted label -> empty set (not malformed).
    """
    text = raw.strip()
    try:
        whole = json.loads(text)
        strict = isinstance(whole, list) and all(isinstance(x, str) for x in whole)
    except json.JSONDecodeError:
        strict = False

    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    match = _LIST_RE.search(text)
    if not match:
        return ParsedOutput((), malformed=True, strict_json=False)
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return ParsedOutput((), malformed=True, strict_json=False)
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        return ParsedOutput((), malformed=True, strict_json=False)

    found = {_normalise(x) for x in items} & _LABEL_SET
    return ParsedOutput(
        tuple(label for label in LABELS if label in found),
        malformed=False,
        strict_json=strict,
    )


def parse_labels(raw: str) -> list[str]:
    """Backward-compatible helper: just the label list."""
    return list(parse_prediction(raw).labels)


def gold_labels(output: Any) -> list[str]:
    """Gold labels from a data row's "output" field (list or JSON string)."""
    if isinstance(output, list):
        items = output
    else:
        parsed = parse_prediction(str(output))
        if parsed.malformed:
            raise ValueError(f"Could not parse gold labels: {output!r}")
        return list(parsed.labels)
    normalised = {_normalise(x) for x in items}
    unknown = normalised - _LABEL_SET
    if unknown:
        raise ValueError(f"Gold labels outside the label set: {unknown}")
    return [label for label in LABELS if label in normalised]


# ---------------------------------------------------------------------------
# 3. Demonstration selection (fixed, aligned IDs)
# ---------------------------------------------------------------------------
def select_covering_ids(
    train_en: Sequence[dict],
    train_target: Sequence[dict],
    id_key: str,
    k: int = 5,
    seed: int = 42,
    min_words: int = 5,
    max_words: int = 20,
    max_labels: int = 3,
    min_single: int = 1,
) -> list:
    """Deterministically choose k aligned training IDs whose labels jointly
    cover all emotions.

    Procedure (report this):
      1. keep IDs present in both train sets with identical gold labels;
      2. keep items with 1..max_labels labels and min_words..max_words English words;
      3. sort candidates by ID; with random.Random(seed), fill the k slots one by one,
         each time choosing at random among candidates that add enough
         not-yet-covered labels to still reach full coverage in the remaining slots;
      4. require at least min_single single-label demonstrations.
    """
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
    chosen: list = []
    uncovered = set(LABELS)
    singles_needed = min_single

    for slot in range(k):
        slots_left = k - slot
        used = {rid for rid, _ in chosen}
        pool = [c for c in candidates if c[0] not in used]

        if slots_left <= singles_needed:
            # Reserve the last slots for single-label items.
            pool = [c for c in pool if len(c[1]) == 1]
            need = -(-len(uncovered) // slots_left)  # ceil
        else:
            # Leave room for the reserved single-label slots.
            multi_slots = slots_left - singles_needed
            need = -(-max(0, len(uncovered) - singles_needed) // multi_slots)
        pool = [c for c in pool if len(c[1] & uncovered) >= need and (c[1] & uncovered or not uncovered)]

        if not pool:
            raise ValueError("Cannot cover all labels with these filters; relax them or raise k.")
        rid, labels = rng.choice(pool)
        chosen.append((rid, labels))
        uncovered -= labels
        if len(labels) == 1:
            singles_needed = max(0, singles_needed - 1)

    if uncovered:
        raise ValueError(f"Labels left uncovered: {sorted(uncovered)}")
    return [rid for rid, _ in chosen]

# Returns the demonstrations as {"id", "text", "labels"} dictionaries.
def build_examples(ids: Sequence, rows: Sequence[dict], id_key: str) -> list[dict]:
    """Turn fixed IDs into demonstrations from ONE language's train set."""
    by_id = {r[id_key]: r for r in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"Five-shot IDs missing from this train set: {missing}")
    return [
        {"id": i, "text": by_id[i]["input"], "labels": gold_labels(by_id[i]["output"])} for i in ids
    ]


# ---------------------------------------------------------------------------
# 4. Generation (greedy decoding)
# ---------------------------------------------------------------------------
DEFAULT_GENERATION = {"max_new_tokens": 64, "batch_size": 8, "chat_template_kwargs": {}}


def to_model_input(
    tokenizer, message: str, chat_template_kwargs: dict | None = None
) -> tuple[str, bool]:
    """Wrap the message in the model's chat template (single user turn).

    Returns (prompt_string, used_chat_template). Models without a chat template
    get the raw message, which already ends with "Labels:".
    """
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            tokenize=False,
            add_generation_prompt=True,
            **(chat_template_kwargs or {}),
        )
        return prompt, True
    return message, False


def generate(
    model,
    tokenizer,
    prompts: Sequence[str],
    used_chat_template: bool,
    max_new_tokens: int = 64,
    batch_size: int = 8,
) -> list[str]:
    """Greedy decoding in left-padded batches. batch_size=1 gives the most
    reproducible results (padding can change bf16 numerics very slightly)."""
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    outputs: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = list(prompts[start : start + batch_size])
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                # The chat template already inserts BOS/special tokens.
                add_special_tokens=not used_chat_template,
            ).to(model.device)
            generated = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = generated[:, enc["input_ids"].shape[1] :]
            outputs += tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return outputs


def predict_with_prompt(
    model,
    tokenizer,
    text: str,
    prompt_type: str = "zero_shot",
    examples=None,
    config: dict | None = None,
    variant: str = DEFAULT_VARIANT,
) -> str:
    """Single-example helper, kept for notebook compatibility."""
    gen = {**DEFAULT_GENERATION, **((config or {}).get("generation", {}))}
    message = build_prompt(text, prompt_type, examples, variant)
    prompt, used = to_model_input(tokenizer, message, gen.get("chat_template_kwargs"))
    return generate(model, tokenizer, [prompt], used, gen["max_new_tokens"], 1)[0]
