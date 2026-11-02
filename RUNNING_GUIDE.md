# Running guide — from setup to final results

This guide describes the current `LLM_sentiment_classification` workflow on the University of Groningen's Hábrók cluster. Run all commands from the **repository root**, unless stated otherwise.

The project compares **Qwen2.5-3B-Instruct** (prompting and PEFT) against **EuroLLM-9B-Instruct** (prompting only) for eight-label emotion classification in English and an assigned language (`de` or `sv`). The repository's YAML files currently select **Swedish (`sv`)** for data preparation and prompting; German is supported as a separate target-language experiment.

**Important rules:** Use TRAIN for training and five-shot demonstrations, DEV for checkpoint/prompt selection, and TEST only after all choices have been frozen. Do not mix results from different configurations or treat an old 8-bit adapter as a 4-bit QLoRA adapter.

## 1. Start from the repository root

```bash
cd /path/to/LLM_sentiment_classification
module load Python/3.12.3-GCCcore-13.3.0
mkdir -p outputs/jobs jobs/logs
```

Use the real path for your Git checkout. Working on `/scratch` is preferable if the repository's `.venv` would otherwise occupy your limited home-directory quota.

### 1.1 Configure caches on scratch — required on Hábrók

**Do this before any `uv sync` or `uv run` command.** The previous `No space left on device` failure occurred because uv downloaded PyTorch/CUDA and `jedi` into `/home2/.../.cache/uv`.

```bash
export UV_CACHE_DIR="/scratch/$USER/llm/uv_cache"
export TMPDIR="/scratch/$USER/llm/tmp"
export HF_HOME="/scratch/$USER/llm/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"

mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$HF_HOME"
uv cache dir
```

The last command should print a path under `/scratch`, **not** `/home2`.

If your repository is located in your *home directory*, also keep its Python environment on scratch:

```bash
export UV_PROJECT_ENVIRONMENT="/scratch/$USER/llm/llm_project_venv"
```

Set `UV_PROJECT_ENVIRONMENT` in **both your terminal and every Slurm job** if you use it. If your repository and `.venv` already live on scratch, you may omit it.

### 1.2 Install dependencies once

```bash
uv sync --locked --no-dev
uv run --no-dev --no-sync python -c "import llm_project, torch, transformers, peft, datasets; print('Environment OK')"
```

`--no-dev` excludes notebook-only development dependencies such as `ipykernel` and `jedi`. `--no-sync` tells later commands to use the installed environment without downloading or reinstalling dependencies. The tracked `uv.lock` fixes package versions.

To run Ruff or a Jupyter kernel later, install the development dependencies separately with `uv sync --locked`. **Do not do this inside GPU training jobs.**

If your home quota is still full, first verify no running job is using the old cache, then optionally clean only the **old home cache**:

```bash
UV_CACHE_DIR="$HOME/.cache/uv" uv cache clean
```

This does not delete project source files or saved adapters; packages may need to be fetched again if another environment depended on that cache.

### 1.3 Make all Slurm jobs use the same environment

Check the four job files:

```text
jobs/prepare_data.sh
jobs/peft_jobs.sh
jobs/eval_peft.sh
jobs/run_prompting.sh
```

Immediately **after** `module load Python/3.12.3-GCCcore-13.3.0` and **before** the `uv` invocation in **each** file, include:

```bash
# keep model downloads, python downloads and temporary files on scratch
export HF_HOME="/scratch/$USER/llm/hf_cache"
export UV_CACHE_DIR="/scratch/$USER/llm/uv_cache"
export TMPDIR="/scratch/$USER/llm/tmp"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"

# uncomment only if the project virtual environment is on scratch separately
# export UV_PROJECT_ENVIRONMENT="/scratch/$USER/llm/llm_project_venv"
```

Change every job's Python command from:

```bash
uv run --locked python scripts/...
```

to:

```bash
uv run --locked --no-dev --no-sync python scripts/...
```

Keep the **existing script name and arguments** after `scripts/...`; change only the uv options. Do not add `--no-dev` or `--no-sync` to `#SBATCH` directives. When a job starts, the environment must already exist from Step 1.2.

For model loading, the cache under `HF_HOME` must contain both model checkpoints. If downloads are required, fetch them using an internet-enabled, cluster-approved node before turning on offline mode. Do not set `HF_HUB_OFFLINE=1` until the models have been cached.

## 2. Check configurations and Python files

The repository separates configuration, implementation and execution:

| Location | Responsibility |
|---|---|
| `configs/data.yaml` | Target language, OPUS/XED input and splitting seed |
| `configs/peft.yaml` | Base Qwen model, LoRA/QLoRA settings, training hyperparameters |
| `configs/prompting.yaml` | Qwen/EuroLLM models, prompting language, generation and five-shot IDs |
| `src/llm_project/data_preparation/` | Dataset matching, cleaning, splitting and saving |
| `src/llm_project/peft_training.py` | Train and save adapters |
| `src/llm_project/peft_evaluation.py` | Generate PEFT predictions and calculate their metrics |
| `src/llm_project/prompting.py` | Prompt templates, generation and demonstration selection |
| `src/llm_project/metrics.py` | **Shared parsing and metrics for prompting and PEFT** |
| `scripts/` | Commands that start each workflow |
| `jobs/` | Cluster resource requests |

**Changes from the older workflow:** `peft_inference.py` is not needed, `scripts/rescore_predictions.py` is not needed for *new* evaluations, and `src/llm_project/prompt_evaluation.py` has been replaced by the shared `metrics.py` module. Both prompting and PEFT calculate scores immediately after generation, using the same parser and metric implementation. `analyze_results.py` only consolidates already-scored results into report tables and plots.

Check for outdated references before submitting jobs:

```bash
# no output is expected for removed modules
rg 'peft_inference|rescore_predictions|prompt_evaluation' src scripts jobs tests

uv run --no-dev --no-sync python -m compileall -q src scripts
uv run --no-dev --no-sync python -m unittest discover -s tests -v
```

If your repository still contains old imports, correct them to import `gold_labels`, `parse_prediction`, `compute_metrics` and `confusion_matrices` from `llm_project.metrics` before deleting old modules.

## 3. Prepare the aligned dataset (CPU)

Check `configs/data.yaml`. The current configuration uses `target_language: sv`, OPUS OpenSubtitles v2018, a 70/20/10 split and seed 40.

```bash
sbatch jobs/prepare_data.sh
squeue -u "$USER"
```

After the job has completed successfully, check:

```bash
ls -lh data/processed/sv/
```

Expected files include:

```text
data/processed/sv/
├── train_en.jsonl
├── train_sv.jsonl
├── train_bilingual.jsonl
├── dev_en.jsonl
├── dev_sv.jsonl
├── dev_bilingual.jsonl
├── test_en.jsonl
├── test_sv.jsonl
├── test_bilingual.jsonl
└── split_manifest.json
```

Data extraction requires access to the source files and may require network access. If downloads from compute nodes fail, use the transfer/download procedure permitted by Hábrók.

**Alignment limitation:** The implementation builds IDs using OPUS document/sentence information. Before making a strict compliance claim in the report, verify their provenance against the **original XED alignment identifiers** required in the course brief. Also check for duplicate lines across splits.

To prepare German, set `target_language: de` in `configs/data.yaml`, submit and **finish** a separate preparation job, then restore `sv` if needed. **Never modify a shared YAML file while a job using it is pending or running**, because it reads the file at runtime.

If the correct processed datasets are already present, skip this step. Do not change or regenerate the split midway through experimental comparison.

## 4. Verify the model and PEFT configuration

In `configs/peft.yaml`, confirm the experiment name, model and training settings. The reviewed configuration uses Qwen2.5-3B-Instruct, LoRA targets `q_proj` and `v_proj`, rank 16, alpha 32, dropout 0.05, seed 42, and `peft.method: qlora` with NF4 **4-bit** quantization.

In `configs/prompting.yaml`, confirm that the target language and both models are correct. The configuration uses Qwen2.5-3B-Instruct (primary), EuroLLM-9B-Instruct (second), greedy decoding, and five fixed aligned demonstration IDs.

Do not rename an older 8-bit adapter as 4-bit QLoRA. If you changed the actual training method or prompt format, train a new adapter under a distinct experiment name rather than silently overwriting a previous run.

## 5. Train PEFT adapters (GPU)

`jobs/peft_jobs.sh` maps the six Slurm indices as follows:

| Array index | Target | Training condition |
|---|---|---|
| 0 | German | English-only |
| 1 | German | German-only |
| 2 | German | Bilingual English/German |
| 3 | Swedish | English-only |
| 4 | Swedish | Swedish-only |
| 5 | Swedish | Bilingual English/Swedish |

Submit the jobs **only after** installing the Python environment in Step 1 and preparing the relevant target-language dataset.

```bash
# swedish only
sbatch --array=3-5%2 jobs/peft_jobs.sh

# german only
sbatch --array=0-2%2 jobs/peft_jobs.sh

# both languages, if both processed datasets are ready
sbatch --array=0-5%2 jobs/peft_jobs.sh
```

Check the job statuses:

```bash
squeue -u "$USER"
sacct -j <JOB_ID> --format=JobID,State,Elapsed,ExitCode
```

Look for `COMPLETED` with exit code `0:0` and a saved adapter in `outputs/models/<target>/<condition>/<experiment>/`.

**If a job fails while downloading torch, Triton, CUDA, or `jedi`**, confirm that the Slurm job exported `UV_CACHE_DIR` to scratch and ran `uv run --no-dev --no-sync`, and that the one-time `uv sync --no-dev` succeeded. Do not keep resubmitting the same job until this is fixed.

If the adapters already exist and match the current dataset/training configuration, skip training.

## 6. Evaluate the PEFT adapters on DEV (GPU)

PEFT evaluation loads the saved adapters, generates predictions on the relevant language partitions and calculates all required metrics immediately using `llm_project.metrics`.

```bash
# swedish
sbatch --array=3-5%2 jobs/eval_peft.sh dev

# german
sbatch --array=0-2%2 jobs/eval_peft.sh dev
```

Each target language produces **five** PEFT evaluation runs:

- English-trained → English and target-language inputs
- Target-trained → target-language input
- Bilingual-trained → English and target-language inputs

After all three array tasks for a target finish, verify:

```bash
ls outputs/predictions/peft/dev/
ls outputs/metrics/peft/dev/
```

There should be one prediction JSONL and one metric JSON per evaluated language/condition. **There is no separate `rescore_predictions.py` step.**

Avoid rerunning completed evaluations unless necessary. `scripts/evaluate_peft.py` supports `--overwrite` if you intentionally need to replace an earlier run. Check that the saved model's `config_used.yaml` matches the adapter you are evaluating.

## 7. Run zero-shot/five-shot prompting on DEV (GPU)

The prompting script loads the original models (no fine-tuning), constructs zero-shot or five-shot prompts, generates responses, parses their JSON emotion labels and immediately calculates the same metrics as PEFT.

For Swedish, check `configs/prompting.yaml` has `language: sv` and exactly five valid training alignment IDs. To inspect or regenerate five-shot candidates:

```bash
uv run --no-dev --no-sync python scripts/run_prompting.py --select-ids
```

Put the chosen five IDs into `prompting.five_shot_ids` and **freeze them** before DEV comparisons. The same IDs are used for the English and target-language sides; the subtitle language changes but the aligned examples do not.

For a small **GPU** smoke test, submit through Slurm; the results go to a separate `dev_smoke` folder:

```bash
sbatch jobs/run_prompting.sh dev --limit 20 --models primary
```

Run full DEV prompting and wording sensitivity on both models:

```bash
sbatch jobs/run_prompting.sh dev --sensitivity --models primary --skip-existing
sbatch jobs/run_prompting.sh dev --sensitivity --models second --skip-existing
```

The DEV sensitivity analysis tests four instruction variants for zero- and five-shot prompting, for both models and both evaluation languages. Files are written under:

```text
outputs/predictions/prompting/dev/
outputs/metrics/prompting/dev/
```

The stored metric JSON files include Micro F1, sample Jaccard, Macro F1, Micro precision/recall, Hamming loss, exact match, per-emotion scores, malformed-output rate, and diagnostic label-cardinality statistics.

## 8. Select prompt wording using DEV

Only after **all DEV prompting runs are complete**:

```bash
uv run --no-dev --no-sync python scripts/analyze_results.py --target sv --split dev --select
```

Replace `sv` with `de` for a German experiment. This selects a prompt variant separately for each model and shot condition using the **mean DEV Micro F1 across English and target-language inputs**, with sample Jaccard as a tiebreaker. The choices are saved to:

```text
outputs/report/selected_prompt_variants.json
```

The DEV summary also generates `outputs/report/<target>_dev_results.csv`, prompt-sensitivity tables/plots, per-emotion tables/figures, dataset summaries and qualitative comparisons.

**Do not change prompts or demonstration IDs after this step.** TEST must use the saved selection. If an expected DEV run is missing, finish it before selecting.

## 9. Run the held-out TEST evaluation once

Use TEST only when your model choices, prompts, selection procedure and training checkpoints are frozen.

For Swedish:

```bash
# prompting: selected versions only
sbatch jobs/run_prompting.sh test --models primary --skip-existing
sbatch jobs/run_prompting.sh test --models second --skip-existing

# peft: all three trained conditions
sbatch --array=3-5%2 jobs/eval_peft.sh test
```

For German, switch `configs/prompting.yaml` to `language: de` **before submitting** the prompting jobs, set the German five-shot IDs, ensure German DEV selection already exists, then run:

```bash
sbatch jobs/run_prompting.sh test --models primary --skip-existing
sbatch jobs/run_prompting.sh test --models second --skip-existing
sbatch --array=0-2%2 jobs/eval_peft.sh test
```

Wait for all jobs to finish. For each target, the required TEST matrix has **8 prompting runs plus 5 PEFT runs**. Each PEFT evaluation job already writes its metric JSON; **do not run a rescoring script**.

## 10. Generate results for the report

For Swedish:

```bash
uv run --no-dev --no-sync python scripts/analyze_results.py --target sv --split test
```

For German:

```bash
uv run --no-dev --no-sync python scripts/analyze_results.py --target de --split test
```

The analysis collects saved metric files, verifies the complete TEST matrix and writes report-ready tables and figures in `outputs/report/`, including:

```text
<target>_test_results.csv
<target>_test_per_emotion.csv
<target>_test_qualitative.csv
<target>_test_runtime.csv
<target>_test_micro_f1_*.png
<target>_test_per_emotion_*.png
```

The report should compare zero-/five-shot prompting with the three PEFT conditions; discuss English–target gaps, prompt sensitivity, per-emotion behavior, malformed outputs, label-cardinality differences, representative improvements/worsenings and measured computational cost.

Do not generate fabricated results when an experiment has not finished. The optional `--allow-incomplete` flag is for debugging partial runs, **not** for the final report.

## 11. Final report and GitHub reproducibility checks

The LaTeX source is in `report/report.tex`. Fill any verification markers with genuine experiment outputs, add real examples and retain the course brief's eight main sections. The required report length is **8–12 pages excluding references and appendices**.

Optional local compilation:

```bash
cd report
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
cd ..
```

Before pushing, from the project root:

```bash
uv run --no-dev --no-sync python -m compileall -q src scripts
uv run --no-dev --no-sync python -m unittest discover -s tests -v
uv lock --check
git status --short
git check-ignore .env data/raw/example.tsv data/processed/sample.jsonl outputs/models/weights.bin outputs/jobs/job.out
```

Track Python code, Slurm scripts, YAML settings, README, this guide, tests, `pyproject.toml`, `uv.lock`, and the LaTeX source. **Do not commit** secrets, `.env`, model weights, raw/processed datasets, cached models, job logs, or generated prediction/metric artifacts.

## Troubleshooting

**`No space left on device` / `jedi` / CUDA download errors** — The uv cache still points to home, or the job is trying to sync its environment. Set `UV_CACHE_DIR` and `TMPDIR` under `/scratch`, run `uv sync --locked --no-dev` once, and use `uv run --no-dev --no-sync` in all jobs.

**`ModuleNotFoundError: llm_project`** — Run from the repository root and check that `uv sync --locked --no-dev` succeeded. Confirm the job uses the same virtual environment as your login shell.

**`No such file or directory: outputs/jobs/...`** — Create `outputs/jobs` **before** submitting. Slurm opens its log paths before your script begins.

**Hugging Face model download fails in the job** — Cache the model on an approved internet-enabled node first. All jobs must share the same `HF_HOME`.

**Missing adapter** — Check that the matching target/condition/experiment training task finished successfully and that the `configs/peft.yaml` experiment name matches the saved directory (or pass `--experiment` when supported).

**Duplicate evaluation output** — Existing prediction files are protected against accidental re-evaluation. Use `--skip-existing` for prompting and `--overwrite` only when intentionally replacing a PEFT result.

**Outdated imports** — Remove any references to `peft_inference`, `rescore_predictions` or `prompt_evaluation` after switching to the shared `llm_project.metrics` module.

**Results missing in analysis** — Confirm all JSON metrics exist in `outputs/metrics/{prompting,peft}/{dev,test}/` and are for the correct target language. DEV selection must be completed before TEST prompting.
