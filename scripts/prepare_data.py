from pathlib import Path

from llm_project.data_preparation import prepare_directory


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    print("Preparing XED data...")

    prepare_directory(
        input_dir=RAW_DATA_DIR,
        output_dir=PROCESSED_DATA_DIR,
    )

    print("Data preparation complete.")


if __name__ == "__main__":
    main()