"""Match XED labels with OPUS alignments and create fixed dataset splits."""

import csv
import json
import random
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

LANGUAGES = ("de", "sv")
SEED = 42
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
RAW = Path("data/raw")
OUT = Path("data/processed")


def clean(text):
    return " ".join(text.split())


def labels_from_xed(text):
    # xed uses 0 for neutral and 1 to 8 for emotions
    values = [value.strip() for value in text.split(",")]
    valid = {str(number) for number in range(9)}
    if not values or any(value not in valid for value in values):
        raise ValueError(f"invalid XED labels: {text!r}")
    if "0" in values and len(values) > 1:
        raise ValueError(f"neutral cannot be combined with emotions: {text!r}")
    return [name for index, name in enumerate(EMOTIONS, 1) if str(index) in values]


def read_xed(lang):
    path = RAW / f"{lang}-projections.tsv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}, run bash get_data.sh")

    labels = {}
    conflicts = set()
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{number}: expected 2 columns, found {len(parts)}\n"
                    "the XED annotations may be corrupted or mixed with OPUS data, re-download them"
                )
            sentence, raw = clean(parts[0]), parts[1]
            try:
                emotions = labels_from_xed(raw)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            if not sentence:
                continue
            if sentence in labels and labels[sentence] != emotions:
                conflicts.add(sentence)
            labels[sentence] = emotions

    for sentence in conflicts:
        del labels[sentence]
    print(f"{lang}: {len(labels)} unique XED texts, {len(conflicts)} conflicting texts removed")
    return labels


def make_candidates(lang):
    xed = read_xed(lang)
    files = (RAW / f"en-{lang}.en", RAW / f"en-{lang}.{lang}", RAW / f"en-{lang}.ids")
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"missing {path}, run bash get_data.sh first")

    matches = {}
    checked = 0
    skipped = 0
    with (
        files[0].open(encoding="utf-8") as en_file,
        files[1].open(encoding="utf-8") as target_file,
        files[2].open(encoding="utf-8") as ids_file,
    ):
        for number, fields in enumerate(zip_longest(en_file, target_file, ids_file), 1):
            if any(field is None for field in fields):
                raise ValueError(
                    f"OPUS text and ID files have different row counts near line {number}\n"
                    "inspect the first five rows of each export before continuing"
                )

            en, target, identifier = (clean(field) for field in fields)
            if not en and not target and not identifier:
                continue
            # skip incomplete aligned pairs
            if not en or not target or not identifier:
                skipped += 1
                continue

            checked += 1
            if target in xed:
                # retain source ids so the same aligned record cannot cross partitions
                matches[target].add((en, identifier))

    path = RAW / f"{lang}-candidates.tsv"
    accepted = 0
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["alignment_id", "en", "target", "labels"])
        for target, possibilities in sorted(matches.items()):
            # require exactly one candidate alignment for each xed subtitle text
            # repeated dialogue with different ids cannot be safely resolved by text alone
            if len(possibilities) != 1:
                continue
            en, identifier = next(iter(possibilities))
            writer.writerow([identifier, en, target, ",".join(xed[target])])
            accepted += 1

    ambiguous = sum(len(pairs) != 1 for pairs in matches.values())
    print(f"checked {checked} OPUS rows")
    print(f"wrote {accepted} candidates to {path}")
    print(f"skipped {skipped} incomplete OPUS alignments")
    print(f"skipped {ambiguous} XED texts with ambiguous OPUS matches")
    print("the IDs are from OPUS and have not been verified as original XED alignment IDs")
    print("for strict XED provenance create a checked <lang>-verified.tsv")


def read_rows(lang):
    path = RAW / f"{lang}-candidates.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}, create candidates and verify original XED IDs first"
        )

    xed = read_xed(lang)
    rows = []
    seen = {}
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if not {"alignment_id", "en", "target", "labels"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} needs alignment_id, en, target, labels columns")
        for number, entry in enumerate(reader, 2):
            identifier = clean(entry["alignment_id"] or "")
            en, target = clean(entry["en"] or ""), clean(entry["target"] or "")
            raw_labels = [
                label.strip().lower()
                for label in (entry["labels"] or "").split(",")
                if label.strip()
            ]
            ordered_labels = [label for label in EMOTIONS if label in raw_labels]
            if not identifier or not en or not target:
                raise ValueError(f"missing ID or sentence at {path}:{number}")
            if len(raw_labels) != len(set(raw_labels)) or set(raw_labels) != set(ordered_labels):
                raise ValueError(f"invalid labels at {path}:{number}")
            if target not in xed or xed[target] != ordered_labels:
                raise ValueError(f"labels disagree with the XED projection at {path}:{number}")

            row = {
                "id": identifier,
                "en": en,
                "target": target,
                "labels": ordered_labels,
            }
            if identifier in seen:
                if seen[identifier] != row:
                    raise ValueError(f"conflicting examples share alignment ID {identifier}")
                continue
            seen[identifier] = row
            rows.append(row)
    if len(rows) < 10:
        raise ValueError(f"only {len(rows)} examples found, need at least 10")
    return rows, path


def split_language(lang, force=False):
    rows, source = read_rows(lang)
    folder = OUT / lang
    if (folder / "train.jsonl").exists() and not force:
        raise FileExistsError(f"{folder} already has a split, use --force to overwrite")

    # grouping duplicates before splitting prevents English/target text leakage
    parents = list(range(len(rows)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen_keys = {}
    for index, row in enumerate(rows):
        for key in (("id", row["id"]), ("en", row["en"]), (lang, row["target"])):
            if key in seen_keys:
                parents[root(index)] = root(seen_keys[key])
            else:
                seen_keys[key] = index

    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[root(index)].append(row)
    units = list(groups.values())
    if len(units) < 10:
        raise ValueError("too few independent alignment groups for 70/20/10 splitting")

    y = np.array(
        [
            [int(any(label in row["labels"] for row in group)) for label in EMOTIONS]
            for group in units
        ]
    )
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)  # type: ignore
    train_dev, test = next(splitter.split(np.zeros(len(units)), y))
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=2 / 9, random_state=SEED)  # type: ignore
    train, dev = next(splitter.split(np.zeros(len(train_dev)), y[train_dev]))
    partitions = {"train": train_dev[train], "dev": train_dev[dev], "test": test}

    folder.mkdir(parents=True, exist_ok=True)
    info = {
        "seed": SEED,
        "source_file": str(source),
        "source_examples": len(rows),
    }
    for name, indices in partitions.items():
        examples = sorted((row for i in indices for row in units[i]), key=lambda row: row["id"])
        if not examples:
            raise ValueError(f"empty {name} split")
        with (folder / f"{name}.jsonl").open("w", encoding="utf-8") as file:
            for row in examples:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        info[name] = {
            "examples": len(examples),
            "groups": len(indices),
            "labels": dict(Counter(label for row in examples for label in row["labels"])),
        }
        print(f"{lang} {name}: {len(examples)} examples from {len(indices)} groups")

    # select five shot examples
    training_examples = [row for i in partitions["train"] for row in units[i]]
    random.Random(SEED).shuffle(training_examples)
    shots = []
    covered = set()
    while len(shots) < 5 and training_examples:
        index = max(
            range(len(training_examples)),
            key=lambda i: len(set(training_examples[i]["labels"]) - covered),
        )
        row = training_examples.pop(index)
        shots.append(row)
        covered.update(row["labels"])
    if len(shots) != 5:
        raise ValueError("fewer than five training examples")
    (folder / "five_shot.json").write_text(
        json.dumps(shots, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    info["five_shot_alignment_ids"] = [row["id"] for row in shots]
    (folder / "split_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
