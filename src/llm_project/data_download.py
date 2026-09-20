from pathlib import Path
from urllib.request import urlretrieve


DATA_URLS = {
    "english_projections": (
        "https://raw.githubusercontent.com/Helsinki-NLP/XED/"
        "master/Projections/en-projections.tsv"
    ),
    "english": (
        "https://raw.githubusercontent.com/Helsinki-NLP/XED/"
        "master/AnnotatedData/en-annotated.tsv"
    ),
    "swedish": (
        "https://raw.githubusercontent.com/Helsinki-NLP/XED/"
        "master/Projections/sv-projections.tsv"
    ),
    "german": (
        "https://raw.githubusercontent.com/Helsinki-NLP/XED/"
        "master/Projections/de-projections.tsv"
    ),
    "nederlands": (
        "https://raw.githubusercontent.com/Helsinki-NLP/XED/"
        "master/Projections/nl-projections.tsv"
    ),
}


def download_xed_data(
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """
    Download XED language files.

    Parameters
    ----------
    output_dir:
        Directory in which the downloaded files are stored.
    overwrite:
        Download files again if they already exist.

    Returns
    -------
    dict[str, Path]
        Mapping from language names to downloaded file paths.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files = {}

    for language, url in DATA_URLS.items():
        filename = Path(url).name
        destination = output_dir / filename

        if destination.exists() and not overwrite:
            print(f"Skipping {language}: already exists.")
        else:
            print(f"Downloading {language} -> {destination}")
            urlretrieve(url, destination)

        downloaded_files[language] = destination

    return downloaded_files