# Single-Model Prediction

This directory contains the prompting-based inference pipeline for evaluating one causal language model on the multilingual emotion-classification data. The pipeline supports zero-shot and fixed five-shot prompting for German (`de`) and Swedish (`sv`) data.

## Where Everything Is

From the repository root:

```text
llm_project/
├── configs/
│   └── prompting.yaml
├── data/processed/
│   ├── de/
│   │   ├── train_en.jsonl
│   │   ├── train_de.jsonl
│   │   ├── dev_en.jsonl
│   │   ├── dev_de.jsonl
│   │   └── test_*.jsonl
│   └── sv/
│       ├── train_en.jsonl
│       ├── train_sv.jsonl
│       ├── dev_*.jsonl
│       └── test_*.jsonl
├── outputs/predictions/prompting/
│   └── <zero_shot|five_shot>/
├── scripts/
│   └── run_prompting.py
└── src/llm_project/single_model_prediction/
    ├── evaluation_metrics.py
    ├── prompt_formatting.py
    ├── prompt_model.py
    └── README.md
```

The package modules are reusable implementation code. The complete experiment is started through `scripts/run_prompting.py`, either directly or through `jobs/prompting_jobs.sh` on the cluster.

## File Responsibilities

### `prompt_formatting.py`

Builds prompts and validates model responses.

- `fetch_fixed_five_shot_examples()` selects the configured training examples and preserves the order of their alignment IDs.
- `prepare_prompt_zero_shot()` creates a prompt containing only the instruction and new text.
- `prepare_prompt_five_shot()` adds the selected demonstrations before the new text.
- `check_output_formatting()` parses the model output as JSON, keeps valid emotion labels, lowercases them, and removes duplicates.

The permitted labels are defined in `src/llm_project/constants.py`:

```text
anger, anticipation, disgust, fear, joy, sadness, surprise, trust
```

### `prompt_model.py`

Owns model loading and inference.

- `load_inference_model()` loads the tokenizer and causal language model from the configured Hugging Face model ID and puts the model in evaluation mode.
- `_generate_prediction()` performs deterministic generation with sampling disabled and returns only newly generated tokens.
- `run_inference()` chooses the zero-shot or five-shot path, generates one prediction per evaluation row, normalizes the prediction, and returns result dictionaries.

Each result contains:

- `alignment_id`
- `input_prompt`
- `prediction` - raw model text
- `normalized_prediction` - validated list of emotion labels
- `true_labels` - labels from the processed dataset

### `evaluation_metrics.py`

Evaluates the DataFrame produced by `run_inference()`.

It reports:

- Primary: micro F1 and sample Jaccard
- Secondary: macro F1, micro precision, micro recall, and Hamming loss
- Diagnostic: exact-match ratio, malformed-output rate, and per-emotion scores

Malformed output is detected from the raw `prediction` column. Metrics use `normalized_prediction` for model labels and `true_labels` for reference labels.

## End-to-End Flow

```text
configs/prompting.yaml
        |
        v
scripts/run_prompting.py
        |
        +--> data/processed/<language>/<split>_<en|language>.jsonl
        |
        +--> prompt_model.load_inference_model()
        |
        +--> run_inference()
                 |
                 +--> prompt_formatting.py
                 |      zero-shot or five-shot prompt
                 |
                 +--> model.generate()
                 |
                 +--> check_output_formatting()
        |
        +--> outputs/predictions/prompting/<mode>/
               <split>_predictions.csv
               <split>_metrics.json
```

## Running the Pipeline

Run from the repository root so the package and project paths resolve correctly:

```bash
uv run python scripts/run_prompting.py \
  --language de \
  --target english \
  --split dev \
  --mode zero_shot
```

Available arguments:

- `--language`: `de` or `sv`; selects the processed-data directory.
- `--target` (also accepted as `--condition`): `english`, `de`, or `sv`; selects the input-language files.
- `--split`: `dev` or `test`; defaults to `dev`.
- `--mode`: `zero_shot` or `five_shot`.

For example, German target-language data uses:

```bash
uv run python scripts/run_prompting.py \
  --language de \
  --target de \
  --split test \
  --mode five_shot
```

The cluster version of these commands is in `jobs/prompting_jobs.sh`. It supplies combinations of language, target, mode, and the test split to SLURM array jobs.

## Configuration

The model is configured in `configs/prompting.yaml`:

```yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
```

For five-shot mode, the runtime configuration must also provide the five training alignment IDs expected by `run_inference()`:

```yaml
fixed_five_shot_ids:
  - "<alignment-id-1>"
  - "<alignment-id-2>"
  - "<alignment-id-3>"
  - "<alignment-id-4>"
  - "<alignment-id-5>"
```

The IDs must exist in the selected training JSONL file. The current command-line parser controls language, target, split, and mode; these values override the corresponding runtime choices.

## Data Selection

`get_data_sets()` in `scripts/run_prompting.py` uses this mapping:

| CLI values | Files loaded |
|---|---|
| `--language de --target english` | `data/processed/de/*_en.jsonl` |
| `--language de --target de` | `data/processed/de/*_de.jsonl` |
| `--language sv --target english` | `data/processed/sv/*_en.jsonl` |
| `--language sv --target sv` | `data/processed/sv/*_sv.jsonl` |

Each split loads `train`, `dev`, and `test`; only the requested `dev` or `test` DataFrame is evaluated. The training DataFrame is needed for five-shot demonstrations.

## Produced Artifacts

Results are written below:

```text
outputs/predictions/prompting/<mode>/
├── dev_predictions.csv
├── dev_metrics.json
├── test_predictions.csv
└── test_metrics.json
```

The prediction CSV preserves the prompt, raw response, normalized labels, and reference labels for every example. The metrics JSON stores the primary, secondary, and diagnostic metric groups generated by `evaluate_results_dataframe()`.

SLURM logs from cluster runs are written under `outputs/jobs/` according to the job script configuration. Model downloads and cache files are managed by the Hugging Face/Transformers environment and are not part of the prediction artifacts.

## Relevant Files for This Solution

For changing the prediction behavior, start here:

1. `src/llm_project/single_model_prediction/prompt_model.py` - model loading, generation, and mode dispatch.
2. `src/llm_project/single_model_prediction/prompt_formatting.py` - prompt templates, demonstrations, and output validation.
3. `src/llm_project/single_model_prediction/evaluation_metrics.py` - metrics and malformed-output handling.
4. `scripts/run_prompting.py` - CLI arguments, project paths, dataset selection, and artifact writing.
5. `configs/prompting.yaml` - model and five-shot configuration.
6. `src/llm_project/constants.py` - instruction text and permitted emotion labels.
7. `jobs/prompting_jobs.sh` - reproducible SLURM experiment combinations.

When editing path logic in `run_prompting.py`, `PROJECT_ROOT` should resolve to the repository directory:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
```

For `scripts/run_prompting.py`, `parents[1]` is `/scratch/s6866425/llm_project`; `parents[2]` is the parent scratch directory and causes configuration and data lookups to fail.
