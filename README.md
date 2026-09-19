# LLM Sentiment Classification

## Project Structure

```text
LLM_sentiment_classification/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   └── llm_sentiment_classification/
├── scripts/
├── configs/
├── jobs/
├── outputs/
├── pyproject.toml
└── uv.lock
```

### `data/`
Contains the dataset.

- `raw/` – original downloaded XED `.tsv` files
- `processed/` – cleaned and transformed data used for training and evaluation

### `notebooks/`
Contains Jupyter notebooks for:

- data exploration
- small-scale experiments
- debugging
- result analysis and visualization

Notebooks should mainly **use functions from `src/`** instead of containing all project logic.

### `src/llm_sentiment_classification/`
Contains the reusable Python code for the project.

Examples:

- `data_download.py` – download XED data
- `data_preparation.py` – load, clean, and encode the dataset
- `models.py` – model and tokenizer setup
- `peft_training.py` – LoRA / QLoRA training logic
- `evaluation.py` – evaluation metrics

### `scripts/`
Contains Python entry points for complete tasks.

Examples:

```text
download_data.py
prepare_data.py
train_peft.py
```

These scripts call functions from `src/` and can also be executed by hypercomputer jobs.

### `configs/`
Contains experiment settings such as:

- model name
- languages
- LoRA rank / alpha
- learning rate
- batch size
- number of epochs

This allows experiments to be changed without modifying the training code.

### `jobs/`
Contains job files used to submit long-running experiments to the university hypercomputer.

### `outputs/`
Contains generated experiment results.

```text
outputs/
├── models/
├── predictions/
└── metrics/
```

Large generated files are not committed to Git.

### `pyproject.toml` and `uv.lock`
Define and lock the Python dependencies.

After cloning the repository:

```bash
uv sync
```

installs the project environment.

---

## Example Workflow

Example: run a LoRA experiment on English XED data.

```text
1. Download data
   ↓
scripts/download_data.py
   ↓
data/raw/

2. Prepare data
   ↓
scripts/prepare_data.py
   ↓
data/processed/

3. Test the idea in Jupyter
   ↓
notebooks/peft_experiments.ipynb

4. Put reusable logic in src/
   ↓
models.py / peft_training.py

5. Define experiment settings
   ↓
configs/english_lora.yaml

6. Run the complete experiment
   ↓
scripts/train_peft.py

7. Submit long training to the hypercomputer
   ↓
jobs/train_english_lora.sh

8. Save results
   ↓
outputs/

9. Analyse results in Jupyter
   ↓
notebooks/results_analysis.ipynb
```

Main rule:

> **Use notebooks for experimentation and analysis, `src/` for reusable code, `scripts/` for complete runs, and `jobs/` for hypercomputer execution.**

## Python Formatting and Linting

The project uses **Ruff** for Python formatting, linting, and import sorting. Ruff is installed as a development dependency and configured in `pyproject.toml`.

Format all Python code:

```bash
uv run ruff format .
```

Automatically fix safe linting issues and organize imports:

```bash
uv run ruff check --fix .
```

Check that the project is correctly formatted and lint-free without changing files:

```bash
uv run ruff format --check .
uv run ruff check .
```

The shared VS Code configuration in .vscode/settings.json uses Ruff as the default Python formatter and formats Python automatically on save.