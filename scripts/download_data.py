from pathlib import Path

from llm_project.data_download import download_xed_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    print("Downloading XED data...")

    download_xed_data(
        output_dir=RAW_DATA_DIR
    )

    print("Download complete.")


if __name__ == "__main__":
    main()