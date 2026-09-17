from pathlib import Path

import polars as pl


SENTIMENT_MAPPING = {
    "anger": "1",
    "anticipation": "2",
    "disgust": "3",
    "fear": "4",
    "joy": "5",
    "sadness": "6",
    "surprise": "7",
    "trust": "8",
}


def load_xed_file(file_path: str | Path) -> pl.DataFrame:
    """
    Load one raw XED TSV file.
    """

    file_path = Path(file_path)

    return pl.read_csv(
        file_path,
        separator="\t",
        has_header=False,
        new_columns=["sentence", "labels"],
        schema_overrides={"labels": pl.String},
    )


def prepare_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean the XED labels and one-hot encode the eight emotions.
    """

    clean_labels = (
        pl.col("labels")
        .str.strip_chars()
        .str.replace_all(" ", "")
        .str.split(",")
    )

    # Store cleaned labels temporarily.
    df = df.with_columns(
        clean_labels.alias("_clean_labels")
    )

    valid_labels = list(SENTIMENT_MAPPING.values())

    # Check whether unexpected labels exist.
    unknown_labels = (
        df.select(
            pl.col("_clean_labels")
            .list.explode()
            .alias("label")
        )
        .filter(
            ~pl.col("label").is_in(valid_labels)
        )
    )

    if unknown_labels.height > 0:
        unknown = unknown_labels["label"].unique().to_list()

        raise ValueError(
            f"Unexpected XED labels: {unknown}"
        )

    # Create one binary column for every emotion.
    emotion_columns = [
        pl.col("_clean_labels")
        .list.contains(pl.lit(label))
        .cast(pl.Int8)
        .alias(emotion)
        for emotion, label in SENTIMENT_MAPPING.items()
    ]

    return (
        df
        .with_columns(emotion_columns)
        .drop("_clean_labels")
    )


def prepare_file(
    input_file: str | Path,
    output_file: str | Path,
) -> pl.DataFrame:
    """
    Load, preprocess and save one XED file.
    """

    input_file = Path(input_file)
    output_file = Path(output_file)

    df = load_xed_file(input_file)
    df = prepare_data(df)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.write_csv(output_file)

    return df


def prepare_directory(
    input_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """
    Process every TSV file inside a directory.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    input_files = sorted(input_dir.glob("*.tsv"))

    if not input_files:
        raise FileNotFoundError(
            f"No TSV files found in {input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for input_file in input_files:
        output_file = (
            output_dir
            / f"{input_file.stem}_processed.csv"
        )

        print(
            f"Processing {input_file.name} "
            f"-> {output_file.name}"
        )

        prepare_file(
            input_file=input_file,
            output_file=output_file,
        )