import json
from numpy import record
import polars as pl
from pathlib import Path

emotion_col = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

def prepare_data(df: pl.DataFrame) -> pl.DataFrame:
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
    
    # Use pl.lit() to ensure Polars treats values as literal strings
    emotion_expressions = [
        clean_labels
        .list.contains(pl.lit(numeric_value))
        .cast(pl.Int8)
        .alias(emotion_name)
        for emotion_name, numeric_value in sentiment_mapping.items()
    ]

    # add target_text column with the corresponding emotion names in a list format
    target_text = (
        pl.concat_list([
            pl.when(clean_labels.list.contains(pl.lit(numeric_value)))
            .then(pl.lit(f'"{emotion_name}"'))
            .otherwise(None)
            for emotion_name, numeric_value in sentiment_mapping.items()
        ])
        .list.drop_nulls()
        .list.join(", ")
    )

    # for every row, wrap the target_text in square brackets to create a list format
    target_text = (
        pl.concat_str([
            pl.lit("["),
            target_text,
            pl.lit("]"),
        ])
        .alias("target_text")
    )

    return df.with_columns([*emotion_expressions, target_text])
    
    
def format_instruction_pair(subtitle: str, emotions: str) -> dict:
    """Formats the raw text into the instruction-response dictionary structure."""
    prompt = (
        "### Instruction:\n"
        "Classify the emotions expressed in the following text. The permitted emotions are: "
        "anger, anticipation, disgust, fear, joy, sadness, surprise, trust.\n\n"
        "### Subtitle:\n"
        f"{subtitle}\n\n"
        "### Emotions:\n"
        f"{emotions}"
    )
    
    # We store the entire formatted block under a single key called "text"
    return {"text": prompt}

def five_shot_format_instruction_pair(list_of_examples: list, list_of_emotions: str) -> dict:
    """This function will format a five-shot example"""
    prompt = (
        "### Instruction:\n"
        "Classify the emotions expressed in the following text. The permitted emotions are: "
        "anger, anticipation, disgust, fear, joy, sadness, surprise, trust.\n\n"
    )
    
    for i, (subtitle, emotions) in enumerate(list_of_examples):
        prompt += (
            f"### Example {i+1}:\n"
            f"Subtitle: {subtitle}\n"
            f"Emotions: {emotions}\n\n"
        )
    
    prompt += (
        "### New Subtitle:\n"
        f"{list_of_emotions[0]}\n\n"
        "### Emotions:\n"
        f"{list_of_emotions[1]}"
    )
    
    return {"text": prompt}
    

def create_jsonl_file(df: pl.DataFrame, output_path: Path):
    with output_path.open("w", encoding="utf-8") as file:
        for entry in df.select(["sentence", "target_text"]).iter_rows(named=True):
            record = format_instruction_pair(entry["sentence"], entry["target_text"])
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_jsonl_file_single_language(df: pl.DataFrame, output_path: Path):
    create_jsonl_file(df, output_path)

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data_files"
    output_dir = script_dir / "processed_data"
    output_dir.mkdir(exist_ok=True)
    input_files = sorted(data_dir.glob("*.tsv"))
    if not input_files:
        raise FileNotFoundError(f"No TSV files found in {data_dir}")

    processed_data_frames = {}

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
        processed_data_frames[file.stem] = df
        
        joy_rows = df.filter(pl.col("joy") == 1)
        print(f"\n--- {file.name}: Rows containing Joy ---")
        print(joy_rows.head(3))
        
        # save file in processed_data directory
        output_file = output_dir / f"{file.stem}_processed.csv"
        df.write_csv(output_file)
        
        # create new directory withing processed_data for jsonl files
        jsonl_dir = output_dir / "jsonl_files"
        jsonl_dir.mkdir(exist_ok=True)

        # save jsonl file for each language
        jsonl_file = jsonl_dir / f"{file.stem}.jsonl"
        create_jsonl_file_single_language(df, jsonl_file)
        
        # prepare fiveshot and save it to a jsonl file
        five_shot_jsonl_file = jsonl_dir / f"{file.stem}_five_shot.jsonl"
        with five_shot_jsonl_file.open("w", encoding="utf-8") as file:
            # get every 5 rows from the dataframe and create a five-shot example
            for i in range(0, df.height, 5):
                five_shot_examples = df[i:i+5].select(["sentence", "target_text"]).to_numpy().tolist()
                if len(five_shot_examples) < 5:
                    break  # Skip if there are less than 5 examples left
                new_example = df[i+5:i+6].select(["sentence", "target_text"]).to_numpy().tolist()
                if not new_example:
                    break  # Skip if there are no more examples left
                record = five_shot_format_instruction_pair(five_shot_examples, new_example[0])
            file.write(json.dumps(record, ensure_ascii=False) + "\n")