import json
import random
import unicodedata
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import opustools
import pandas as pd
import yaml
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedShuffleSplit,
)

from llm_project.constants import EMOTION_MAP, EMOTIONS, INSTRUCTION, XED_BASE_URL


# load the config .yaml file
def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def download_xed_projection(
    target_language: str,
    output_dir: str | Path,
) -> Path:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{target_language}-projections.tsv"
    destination = output_dir / filename

    if destination.exists():
        print(f"XED file already exists: {destination}")
        return destination

    url = f"{XED_BASE_URL}/{filename}"

    print(f"Downloading XED projection: {filename}")

    urlretrieve(url, destination)

    return destination


def parse_labels(label_string: str) -> list[str]:
    ids = [value.strip() for value in str(label_string).split(",") if value.strip()]

    unknown = set(ids) - set(EMOTION_MAP) - {"0"}

    if unknown:
        raise ValueError(f"Unexpected XED labels: {sorted(unknown)}")

    # neutral is not one of the eight allowed task labels
    if ids == ["0"]:
        return []

    # neutral should not coexist with emotion labels
    if "0" in ids:
        raise ValueError(f"Neutral label mixed with emotion labels: {ids}")

    return [EMOTION_MAP[label_id] for label_id in sorted(ids, key=int)]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = " ".join(text.strip().split())
    return text


def load_xed_projection(
    file_path: str | Path,
) -> pd.DataFrame:

    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["target_sentence", "label_ids"],
        dtype={
            "target_sentence": str,
            "label_ids": str,
        },
    )

    df["labels"] = df["label_ids"].apply(parse_labels)

    # remove neutral-only examples
    df = df[df["labels"].map(len) > 0].copy()
    df["target_key"] = df["target_sentence"].apply(normalize_text)
    df["label_key"] = df["labels"].apply(tuple)

    # detect identical sentences with conflicting emotion labels
    label_counts = df.groupby("target_key")["label_key"].nunique()
    conflicting_keys = set(label_counts[label_counts > 1].index)

    if conflicting_keys:
        print(f"Dropping {len(conflicting_keys)} target sentences with conflicting labels.")
        df = df[~df["target_key"].isin(conflicting_keys)]

    # identical sentence + identical labels -> keep one copy
    df = (
        df.sort_values("target_key")
        .drop_duplicates(
            subset=["target_key"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    # binary columns used for stratification/evaluation
    for emotion in EMOTIONS:
        df[emotion] = df["labels"].apply(lambda labels: int(emotion in labels))

    return df


def create_opus_pairs(
    source_language: str,
    target_language: str,
    corpus: str,
    release: str,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)

    pair_dir = output_dir / f"{source_language}-{target_language}"
    cache_dir = output_dir / "cache"

    pair_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    source_file = pair_dir / f"{source_language}-{target_language}.{source_language}.txt"
    target_file = pair_dir / f"{source_language}-{target_language}.{target_language}.txt"
    ids_file = pair_dir / f"{source_language}-{target_language}.ids.tsv"

    if source_file.exists() and target_file.exists() and ids_file.exists():
        print("OPUS pair files already exist.")
        return source_file, target_file, ids_file

    print(f"Extracting OPUS {source_language}-{target_language} alignments...")

    reader = opustools.OpusRead(
        directory=corpus,
        source=source_language,
        target=target_language,
        release=release,
        preprocess="raw",
        # only keep 1-to-1 sentence alignments
        src_range="1",
        tgt_range="1",
        leave_non_alignments_out=True,
        # two output files instead of one TSV file
        write=[
            str(source_file),
            str(target_file),
        ],
        write_mode="moses",
        # metadata for the exact same aligned rows
        write_ids=str(ids_file),
        download_dir=str(cache_dir),
        suppress_prompts=True,
    )

    reader.printPairs()

    return source_file, target_file, ids_file


def load_opus_pairs(
    source_file: str | Path,
    target_file: str | Path,
    ids_file: str | Path,
) -> pd.DataFrame:
    source_file = Path(source_file)
    target_file = Path(target_file)
    ids_file = Path(ids_file)

    # read one sentence per line
    with source_file.open("r", encoding="utf-8") as file:
        source_sentences = [line.rstrip("\n") for line in file]

    with target_file.open("r", encoding="utf-8") as file:
        target_sentences = [line.rstrip("\n") for line in file]

    # the ID file really is tabular:
    # source document, target document,
    # source sentence ID, target sentence ID, attribute
    ids = pd.read_csv(
        ids_file,
        sep="\t",
        header=None,
        names=[
            "source_document",
            "target_document",
            "source_sentence_id",
            "target_sentence_id",
            "attribute",
        ],
        dtype=str,
        keep_default_na=False,
    )

    # all three outputs must contain exactly the same number of rows
    if not (len(source_sentences) == len(target_sentences) == len(ids)):
        raise ValueError(
            "OPUS source sentences, target sentences, "
            "and ID rows have different lengths: "
            f"source={len(source_sentences)}, "
            f"target={len(target_sentences)}, "
            f"ids={len(ids)}"
        )

    print(f"Loaded {len(ids):,} OPUS aligned sentence pairs.")

    df = ids.copy()

    df["english_sentence"] = source_sentences
    df["target_sentence"] = target_sentences

    # unique ID for the English side of the aligned subtitle
    df["alignment_id"] = df["source_document"] + "#" + df["source_sentence_id"]

    df["english_key"] = df["english_sentence"].apply(normalize_text)

    df["target_key"] = df["target_sentence"].apply(normalize_text)

    return df


def build_matched_pairs(
    xed: pd.DataFrame,
    opus: pd.DataFrame,
) -> pd.DataFrame:

    # only keep OPUS sentences that occur in XED
    matched = opus.merge(
        xed[
            [
                "target_key",
                "labels",
                *EMOTIONS,
            ]
        ],
        on="target_key",
        how="inner",
    )

    print(f"Initial XED <-> OPUS matches: {len(matched)}")

    # now, handle ambiguous cases because it can happen that if exactly the same target sentence
    # occurs at multiple OPUS alignment IDs, we can't know which occurrence corresponds to XED from text alone
    # so, drop those ambiguous cases

    # tODO: maybe filter by semantic similarity using embeddings later
    counts = matched.groupby("target_key")["alignment_id"].nunique()

    ambiguous = set(counts[counts > 1].index)
    if ambiguous:
        print(f"Dropping {len(ambiguous)} ambiguous target subtitle lines.")
        matched = matched[~matched["target_key"].isin(ambiguous)]

    matched = matched.sort_values("alignment_id").drop_duplicates(
        subset=["target_key"],
        keep="first",
    )

    # also, we need to guarantee that an identical English subtitle can't occur in multiple partitions
    # so, keep one deterministic occurence
    before = len(matched)
    matched = matched.drop_duplicates(
        subset=["english_key"],
        keep="first",
    )
    removed_english_duplicates = before - len(matched)
    if removed_english_duplicates:
        print(f"Dropped {removed_english_duplicates} duplicate English subtitle lines.")

    return matched.reset_index(drop=True)


def create_fixed_split(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
) -> pd.DataFrame:

    # validity check of split distribution
    if not np.isclose(
        train_ratio + dev_ratio + test_ratio,
        1.0,
    ):
        raise ValueError("Train/dev/test ratios must sum to 1.")

    # stable order before random splitting
    df = df.sort_values("alignment_id").reset_index(drop=True)

    X = np.zeros((len(df), 1))
    y = df[EMOTIONS].to_numpy()

    # first, determine the training set (like training, temporary)
    splitter_1 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=dev_ratio + test_ratio,  # type: ignore
        random_state=seed,
    )

    train_idx, temp_idx = next(splitter_1.split(X, y))
    train = df.iloc[train_idx].copy()
    temp = df.iloc[temp_idx].copy()

    # now, split up the temporary set into dev and test set
    temp_test_fraction = test_ratio / (dev_ratio + test_ratio)
    splitter_2 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=temp_test_fraction,  # type: ignore
        random_state=seed,
    )

    X_temp = np.zeros((len(temp), 1))
    y_temp = temp[EMOTIONS].to_numpy()

    dev_idx, test_idx = next(splitter_2.split(X_temp, y_temp))
    dev = temp.iloc[dev_idx].copy()
    test = temp.iloc[test_idx].copy()

    train["split"] = "train"
    dev["split"] = "dev"
    test["split"] = "test"

    result = pd.concat([train, dev, test], ignore_index=True)

    return result


def validate_split(
    df: pd.DataFrame,
) -> None:

    # every alignment can only belong to exactly one partition
    alignment_counts = df.groupby("alignment_id")["split"].nunique()
    if (alignment_counts > 1).any():
        raise ValueError("Alignment leakage detected.")

    # exact duplicate English lines cannot cross partitions
    english_counts = df.groupby("english_key")["split"].nunique()
    if (english_counts > 1).any():
        raise ValueError("Duplicate English lines occur across partitions.")

    # -----------------------------------------
    # same for target language
    # -----------------------------------------
    # exact duplicate target-language lines cannot cross partitions
    target_counts = df.groupby("target_key")["split"].nunique()
    if (target_counts > 1).any():
        raise ValueError("Duplicate target-language lines occur across partitions.")

    print("\nSplit validation passed.")
    print("\nSplit sizes:")

    print(df["split"].value_counts(normalize=True).sort_index())

    # verify equal distribution of emotions of samples
    print("\nEmotion prevalence:")
    print(df.groupby("split")[EMOTIONS].mean())


def make_sft_records(
    df: pd.DataFrame,
    sentence_column: str,
    language: str,
) -> list[dict]:

    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "alignment_id": row["alignment_id"],
                "language": language,
                "instruction": INSTRUCTION,
                "input": row[sentence_column],
                "output": json.dumps(
                    row["labels"],
                    ensure_ascii=False,
                ),
            }
        )

    return records


def write_jsonl(
    records: list[dict],
    file_path: str | Path,
) -> None:

    file_path = Path(file_path)
    with file_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def balanced_bilingual_records(
    english_records: list[dict],
    target_records: list[dict],
    target_language: str,
    seed: int,
) -> list[dict]:
    """Select one language per aligned TRAIN pair, with equal total size."""

    def index_by_id(records: list[dict], language: str) -> dict[str, dict]:
        indexed = {}
        for record in records:
            if record["language"] != language:
                raise ValueError(f"Expected language {language!r} in training records")
            alignment_id = str(record["alignment_id"])
            if alignment_id in indexed:
                raise ValueError(f"Duplicate alignment ID: {alignment_id}")
            indexed[alignment_id] = record
        return indexed

    english = index_by_id(english_records, "en")
    target = index_by_id(target_records, target_language)

    if not english or english.keys() != target.keys():
        raise ValueError("English and target TRAIN alignment IDs do not match")

    # check matching gold labels before selecting a language
    for alignment_id in english:
        if json.loads(english[alignment_id]["output"]) != json.loads(
            target[alignment_id]["output"]
        ):
            raise ValueError(f"Conflicting labels for alignment ID {alignment_id}")

    rng = random.Random(seed)
    ids = sorted(english)
    rng.shuffle(ids)
    n_english = (len(ids) + 1) // 2

    bilingual = [
        english[alignment_id] if i < n_english else target[alignment_id]
        for i, alignment_id in enumerate(ids)
    ]
    rng.shuffle(bilingual)

    print(
        f"Balanced bilingual TRAIN: {len(bilingual)} examples "
        f"({n_english} EN, {len(bilingual) - n_english} {target_language.upper()})"
    )
    return bilingual


def rebalance_existing_training(
    project_root: str | Path,
    target_language: str,
    seed: int,
) -> None:
    """Rewrite ONLY the bilingual TRAIN file using existing processed TRAIN splits."""
    folder = Path(project_root) / "data" / "processed" / target_language

    def read_jsonl(path: Path) -> list[dict]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run --stage split first.")
        with path.open(encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]

    english = read_jsonl(folder / "train_en.jsonl")
    target = read_jsonl(folder / f"train_{target_language}.jsonl")
    balanced = balanced_bilingual_records(english, target, target_language, seed)

    output = folder / "train_bilingual.jsonl"
    temporary = folder / "train_bilingual.jsonl.tmp"
    write_jsonl(balanced, temporary)
    temporary.replace(output)
    print(f"Updated {output} without modifying TRAIN/DEV/TEST assignments")


def save_processed_data(
    df: pd.DataFrame,
    target_language: str,
    output_dir: str | Path,
    seed: int,
) -> None:

    output_dir = Path(output_dir) / target_language

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # convert list labels into JSON text for CSV
    csv_df = df.copy()
    csv_df["labels"] = csv_df["labels"].apply(
        lambda labels: json.dumps(
            labels,
            ensure_ascii=False,
        )
    )

    # generate full audit file
    csv_df.to_csv(
        output_dir / "all_pairs.csv",
        index=False,
    )

    # esnure concrete mapping that can be used by everyone in the team
    csv_df[["alignment_id", "split"]].to_csv(
        output_dir / "split_assignments.csv",
        index=False,
    )

    # save pair-level partitions
    for split in ["train", "dev", "test"]:
        subset = df[df["split"] == split].copy()

        subset_csv = csv_df[csv_df["split"] == split]
        subset_csv.to_csv(output_dir / f"{split}_pairs.csv", index=False)

        # english version
        english_records = make_sft_records(
            subset, sentence_column=("english_sentence"), language="en"
        )

        # assigned-language version
        target_records = make_sft_records(
            subset, sentence_column=("target_sentence"), language=target_language
        )

        write_jsonl(english_records, output_dir / f"{split}_en.jsonl")
        write_jsonl(target_records, output_dir / f"{split}_{target_language}.jsonl")

        # balance TRAIN once & keep DEV/TEST bilingual files unchanged
        if split == "train":
            bilingual_records = balanced_bilingual_records(
                english_records, target_records, target_language, seed
            )
        else:
            bilingual_records = english_records + target_records

        write_jsonl(bilingual_records, output_dir / f"{split}_bilingual.jsonl")

    # document the split
    manifest = {
        "seed": seed,
        "target_language": target_language,
        "emotions": EMOTIONS,
        "counts": (df["split"].value_counts().to_dict()),
    }

    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


def run_pipeline(
    project_root: str | Path,
    config_path: str | Path,
    stage: str = "all",
) -> None:
    """Download sources, build splits, or rebalance existing TRAIN records."""
    if stage not in ("download", "split", "rebalance", "all"):
        raise ValueError(f"Unknown data preparation stage: {stage}")

    project_root = Path(project_root)
    config = load_config(config_path)
    source_language = config["source_language"]
    target_language = config["target_language"]
    seed = config["split"]["seed"]

    raw_xed_dir = project_root / "data" / "raw" / "xed"
    raw_opus_dir = project_root / "data" / "raw" / "opus"
    processed_dir = project_root / "data" / "processed"

    if stage == "rebalance":
        rebalance_existing_training(project_root, target_language, seed)
        return

    # only the download stage contacts the source servers or extracts OPUS
    if stage in ("download", "all"):
        download_xed_projection(target_language, raw_xed_dir)
        create_opus_pairs(
            source_language=source_language,
            target_language=target_language,
            corpus=config["opus"]["corpus"],
            release=config["opus"]["release"],
            output_dir=raw_opus_dir,
        )
        if stage == "download":
            print("Download stage complete. Existing processed splits were not touched.")
            return

    # split stage reads already downloaded files; it never downloads sources
    xed_file = raw_xed_dir / f"{target_language}-projections.tsv"
    pair_dir = raw_opus_dir / f"{source_language}-{target_language}"
    source_file = pair_dir / f"{source_language}-{target_language}.{source_language}.txt"
    target_file = pair_dir / f"{source_language}-{target_language}.{target_language}.txt"
    ids_file = pair_dir / f"{source_language}-{target_language}.ids.tsv"

    for path in (xed_file, source_file, target_file, ids_file):
        if not path.is_file():
            raise FileNotFoundError(f"Missing source file: {path}. First run --stage download.")

    xed = load_xed_projection(xed_file)
    print(f"XED labelled examples: {len(xed)}")
    opus = load_opus_pairs(source_file, target_file, ids_file)
    pairs = build_matched_pairs(xed, opus)
    print(f"Final matched pairs: {len(pairs)}")

    pairs = create_fixed_split(
        df=pairs,
        seed=seed,
        train_ratio=config["split"]["train"],
        dev_ratio=config["split"]["dev"],
        test_ratio=config["split"]["test"],
    )
    validate_split(pairs)

    # prevent accidentally changing an established split during later reruns
    assignment_file = processed_dir / target_language / "split_assignments.csv"
    if assignment_file.exists():
        existing = pd.read_csv(assignment_file, dtype={"alignment_id": str})
        old_assignments = dict(zip(existing["alignment_id"], existing["split"], strict=True))
        new_assignments = dict(zip(pairs["alignment_id"].astype(str), pairs["split"], strict=True))
        if old_assignments != new_assignments:
            raise ValueError(
                "Existing TRAIN/DEV/TEST assignments differ from the regenerated split. "
                "No files were overwritten. Use --stage rebalance to modify only "
                "the bilingual training data."
            )

    save_processed_data(pairs, target_language, processed_dir, seed)
    print("Data preparation complete.")
