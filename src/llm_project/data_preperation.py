import polars as pl
from pathlib import Path

sentiment_mapping = {
    "anger": "1",
    "anticipation": "2",
    "disgust": "3",
    "fear": "4",
    "joy": "5",
    "sadness": "6",
    "surprise": "7",
    "trust": "8",
}

def prepare_data(df: pl.DataFrame) -> pl.DataFrame:

    
    # Clean whitespace and unseen characters removed. this is logic not data
    clean_labels = (
        pl.col("labels")
        .str.strip_chars()
        .str.replace_all(" ", "")
        .str.split(",")
    )

    unknown_labels = (
        df.select(clean_labels.list.explode().alias("label"))
        .filter(~pl.col("label").is_in(list(sentiment_mapping.values())))
    )
    if unknown_labels.height:
        raise ValueError(f"Unexpected XED labels: {unknown_labels['label'].unique().to_list()}")
    
    # Use pl.lit() to ensure Polars treats the value strictly as a literal string
    expressions = [
        clean_labels
        .list.contains(pl.lit(numeric_value)) # boolean mask for each sentiment
        .cast(pl.Int8)
        .alias(emotion_name)
        for emotion_name, numeric_value in sentiment_mapping.items()
    ]
    
    return df.with_columns(expressions) # for every row in the dataframe, add one hot encoding
    
if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data_files"
    output_dir = script_dir / "processed_data"
    output_dir.mkdir(exist_ok=True)
    input_files = sorted(data_dir.glob("*.tsv"))
    if not input_files:
        raise FileNotFoundError(f"No TSV files found in {data_dir}")

    processed_data = {}

    for file in input_files:
        # Explicit schema override prevents Polars from guessing the labels are integers
        df = pl.read_csv(
            file, 
            separator="\t", 
            has_header=False, 
            new_columns=["sentence", "labels"],
            schema_overrides={"labels": pl.String}
        )
        
        df = prepare_data(df)
        processed_data[file.stem] = df
        
        joy_rows = df.filter(pl.col("joy") == 1)
        print(f"\n--- {file.name}: Rows containing Joy ---")
        print(joy_rows.head(3))
        
        # save file in processed_data directory
        output_file = output_dir / f"{file.stem}_processed.csv"
        df.write_csv(output_file)