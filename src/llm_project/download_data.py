import datasets
import os
import pandas as pd
import urllib.request
from pathlib import Path

os.mkdir("data_files") if not os.path.exists("data_files") else None

languages = {
    "english_projections": "https://raw.githubusercontent.com/Helsinki-NLP/XED/master/Projections/en-projections.tsv",
    "english": "https://raw.githubusercontent.com/Helsinki-NLP/XED/master/AnnotatedData/en-annotated.tsv",
    "swedish": "https://raw.githubusercontent.com/Helsinki-NLP/XED/master/Projections/sv-projections.tsv",
    "german": "https://raw.githubusercontent.com/Helsinki-NLP/XED/master/Projections/de-projections.tsv"
}

print("Downloading data...")
for language, url in languages.items():
    filename = Path(url).name
    data_dir = Path("data_files")
    destination = data_dir / filename
    print(f"Downloading {language} data to {destination}")
    urllib.request.urlretrieve(url, destination)

print("Downloads complete.\n")

# The files are tab-separated (sentence \t labels)
df = pd.read_csv(data_dir / "en-annotated.tsv", sep="\t", header=None, names=["sentence", "labels"])
print(df.head())