# Running guide — German and Swedish

This guide explains how to run the project on the **Hábrók HPC cluster** using Slurm. Submit commands from the repository root. See [README.md](README.md) for the project objective and folder structure.

## 1. Prepare the Python environment

Connect to Hábrók and open the repository:

```bash
cd /scratch/$USER/llm/LLM_sentiment_classification
module load Python/3.12.3-GCCcore-13.3.0
mkdir -p outputs/jobs jobs/logs /scratch/$USER/llm/uv_cache /scratch/$USER/llm/tmp
```

Install dependencies **once**:

```bash
UV_CACHE_DIR="/scratch/$USER/llm/uv_cache" \
TMPDIR="/scratch/$USER/llm/tmp" \
uv sync --locked --no-dev
```

- `--locked` uses the existing `uv.lock` without changing dependencies
- `--no-dev` skips development-only packages
- The two temporary variables put downloads in scratch instead of your limited home directory

**You do not need to export Hugging Face or uv cache variables before submitting Slurm jobs.** Your `jobs/*.sh` scripts already set the necessary environment variables, including `HF_HOME`, and load Python. Check that your local copies of those job scripts still contain these settings.

If you **manually run Python scripts** instead of submitting Slurm jobs, set the cache paths in that terminal first:

```bash
export HF_HOME="/scratch/$USER/llm/hf_cache"
export UV_CACHE_DIR="/scratch/$USER/llm/uv_cache"
export TMPDIR="/scratch/$USER/llm/tmp"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"
```

There is no need to set separate Hugging Face Hub or Xet cache variables: `HF_HOME` is sufficient here.

### Optional `.env` file

The `.env.example` file shows optional local settings. If you need them:

```bash
cp .env.example .env
```

Edit `.env` for your environment and keep secrets out of Git. **Creating `.env` does not automatically load its values** into Slurm jobs or Python scripts; the job scripts handle their own environment settings.


## 2. Download both models once (Hugging Face cache)

Before running the GPU experiments, download both models to **scratch** once. This is a manual, one-time setup step. The Slurm jobs will reuse the same cached files; they do not need to download the models again.

Do this **after** installing the Python environment from Section 1, on a Hábrók node where downloads are permitted.

### Set the cache location for this manual download

```bash
export HF_HOME="/scratch/$USER/llm/hf_cache"
export UV_CACHE_DIR="/scratch/$USER/llm/uv_cache"
export TMPDIR="/scratch/$USER/llm/tmp"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"
unset HF_HUB_OFFLINE
```

- `HF_HOME` stores the Hugging Face models on scratch instead of in your home directory
- `UV_CACHE_DIR` and `TMPDIR` keep uv and temporary files on scratch
- `unset HF_HUB_OFFLINE` allows the initial download to contact Hugging Face

### Download Qwen and EuroLLM

```bash
uv run --locked --no-dev --no-sync python - <<'PYCODE'
from huggingface_hub import snapshot_download

models = [
    "Qwen/Qwen2.5-3B-Instruct",
    "utter-project/EuroLLM-9B-Instruct",
]

for model in models:
    path = snapshot_download(repo_id=model)
    print(f"Cached {model}: {path}")
PYCODE
```

- `--locked` uses the existing lockfile
- `--no-dev` excludes development dependencies
- `--no-sync` uses the already installed environment without installing packages
- `snapshot_download()` downloads a model repository into the cache; existing cached files are reused

If Hugging Face asks for authentication or denies access, check the model's page for access requirements, then log in with a read-access token:

```bash
uv run --locked --no-dev --no-sync python -c 'from huggingface_hub import login; login()'
```

Do not put your Hugging Face token in the repository.

### Verify both models are cached

```bash
uv run --locked --no-dev --no-sync python - <<'PYCODE'
from huggingface_hub import snapshot_download

for model in ["Qwen/Qwen2.5-3B-Instruct", "utter-project/EuroLLM-9B-Instruct"]:
    path = snapshot_download(repo_id=model, local_files_only=True)
    print(f"Found {model}: {path}")
PYCODE
```

`local_files_only=True` checks the local cache without downloading anything. After both models are present, your Slurm jobs can use `HF_HUB_OFFLINE=1` if their scripts enable offline mode.

**Important:** You only need to set these environment variables manually for commands you run directly in a terminal (such as downloading models or running a Python script yourself). Your Slurm job files already set the cache paths. **Do not add separate `HF_HUB_CACHE` or `HF_XET_CACHE` exports.**


## 3. Check the job files

Each Slurm job should invoke the corresponding Python script:

| Slurm job | Python script |
|---|---|
| `jobs/run_data_preparation.sh` | `scripts/run_data_preparation.py` |
| `jobs/run_peft_training.sh` | `scripts/run_peft_training.py` |
| `jobs/run_peft_evaluation.sh` | `scripts/run_peft_evaluation.py` |
| `jobs/run_prompting.sh` | `scripts/run_prompting.py` |
| `jobs/run_results.sh` | `scripts/run_results.py` |

For prompting, `run_prompting.sh` must forward additional arguments using `"$@"`, so flags like `--language` and `--models` reach Python. The commands below assume that its first positional argument is `dev` or `test` and that it passes this to Python as `--split`.

## 4. Prepare the datasets

In `configs/data.yaml`, select the target (`de` or `sv`). Prepare each language once:

```bash
sbatch jobs/run_data_preparation.sh
```

This job creates the processed TRAIN, DEV and TEST datasets. After the first job **finishes**, change `target_language` in `configs/data.yaml` and submit it again for the other target. Do not regenerate the splits after training has started.

Check that both language folders exist:

```bash
ls data/processed/de/
ls data/processed/sv/
```

## 5. Train and evaluate PEFT

The following commands assume the Slurm array maps `0–2` to German and `3–5` to Swedish, with conditions `english`, `target` and `bilingual`. **Check that mapping in your job scripts before submitting.**

Train the adapters:

```bash
sbatch --array=0-2%2 jobs/run_peft_training.sh
sbatch --array=3-5%2 jobs/run_peft_training.sh
```

- `--array=0-2%2` runs the three German conditions, at most two concurrently
- `--array=3-5%2` does the same for Swedish

After training finishes, evaluate on DEV:

```bash
sbatch --array=0-2%2 jobs/run_peft_evaluation.sh dev
sbatch --array=3-5%2 jobs/run_peft_evaluation.sh dev
```

- `dev` selects the development split
- Predictions go to `outputs/predictions/peft/dev/` and metrics to `outputs/metrics/peft/dev/`

## 6. Configure prompting once

There is only **one** prompting configuration: `configs/prompting.yaml`. Check the model roles:

```yaml
models:
  - name: Qwen/Qwen2.5-3B-Instruct
    role: primary
  - name: utter-project/EuroLLM-9B-Instruct
    role: secondary
```

Store the five fixed examples **separately for each language**, under `prompting.five_shot_ids`:

```yaml
prompting:
  id_key: alignment_id
  five_shot_ids:
    de: []  # replace with five German alignment IDs
    sv: []  # replace with five Swedish alignment IDs
  selection:
    k: 5
    seed: 42
    min_words: 5
    max_words: 20
    max_labels: 3
    min_single: 1
```

The lists shown above are placeholders. Keep your existing selection settings and paste in **actual IDs from your TRAIN data**.

To find the five IDs for each target, run these **one-time manual commands** (with the manual cache exports from Section 1):

```bash
uv run --locked --no-dev --no-sync python scripts/run_prompting.py --language de --select-ids
uv run --locked --no-dev --no-sync python scripts/run_prompting.py --language sv --select-ids
```

- `--language de` / `--language sv` chooses the target pair; it **overrides any default `language:` in YAML**
- `--select-ids` prints suitable aligned five-shot examples and exits without model inference

Copy the printed IDs into `five_shot_ids.de` and `five_shot_ids.sv`. Once both lists are filled, **you no longer need to edit `prompting.yaml` when switching languages**.

## 7. Run prompting on DEV

Run both models for German and Swedish:

```bash
sbatch jobs/run_prompting.sh dev --language de --sensitivity --models primary --skip-existing
sbatch jobs/run_prompting.sh dev --language de --sensitivity --models secondary --skip-existing
sbatch jobs/run_prompting.sh dev --language sv --sensitivity --models primary --skip-existing
sbatch jobs/run_prompting.sh dev --language sv --sensitivity --models secondary --skip-existing
```

- `dev` evaluates the development split
- `--language de/sv` selects the language pair and its five-shot IDs
- `--sensitivity` evaluates all **four prompt versions** rather than only `default`
- `--models primary/secondary` selects Qwen or EuroLLM
- `--skip-existing` skips a run when **both** its predictions and metrics already exist

For each target, there are **32 DEV experiments**: two models × two input languages × two shot settings × four prompt versions. The two targets can now run independently because their datasets, five-shot IDs and experiment filenames are language-specific.

Check counts when the jobs finish:

```bash
find outputs/metrics/prompting/dev -maxdepth 1 -name '*__de__*.json' | wc -l  # 32
find outputs/metrics/prompting/dev -maxdepth 1 -name '*__sv__*.json' | wc -l  # 32
```

Optional smoke test (manual script, inside a GPU allocation):

```bash
uv run --locked --no-dev --no-sync python scripts/run_prompting.py --language sv --split dev --models primary --limit 20
```

- `--split dev` selects DEV when calling the Python script directly
- `--limit 20` processes only 20 examples and writes to a separate `dev_smoke/` folder; do not count these as full evaluations

## 8. Select the best DEV prompts

After all prompting DEV runs are finished:

```bash
sbatch jobs/run_results.sh --target both --split dev --select
```

- `--target both` analyzes German and Swedish
- `--split dev` reads DEV results
- `--select` chooses the highest-scoring prompt wording using DEV Micro F1

This creates two **independent** files:

```text
outputs/report/selected_prompt_variants_de.json
outputs/report/selected_prompt_variants_sv.json
```

It also generates the combined DEV prompt-sensitivity figure. You do not need to rerun any models in this step.

## 9. Run the final TEST evaluations

Only run TEST after DEV selection has finished. **Do not use `--sensitivity` or `--limit` on TEST.**

Prompting TEST:

```bash
sbatch jobs/run_prompting.sh test --language de --models primary --skip-existing
sbatch jobs/run_prompting.sh test --language de --models secondary --skip-existing
sbatch jobs/run_prompting.sh test --language sv --models primary --skip-existing
sbatch jobs/run_prompting.sh test --language sv --models secondary --skip-existing
```

- `test` selects the held-out test split
- `--language` chooses the target, and the code reads its `selected_prompt_variants_de.json` or `_sv.json`
- Only the DEV-selected wording is evaluated for each model and shot setting

PEFT TEST:

```bash
sbatch --array=0-2%2 jobs/run_peft_evaluation.sh test
sbatch --array=3-5%2 jobs/run_peft_evaluation.sh test
```

- `test` selects the held-out split; the array indices select language and training condition

Expected metric files **per language pair**: eight TEST prompting runs and five TEST PEFT runs. Check them:

```bash
find outputs/metrics/prompting/test -maxdepth 1 -name '*__de__*.json' | wc -l  # 8
find outputs/metrics/prompting/test -maxdepth 1 -name '*__sv__*.json' | wc -l  # 8
find outputs/metrics/peft/test -maxdepth 1 -name '*__de__*.json' | wc -l       # 5
find outputs/metrics/peft/test -maxdepth 1 -name '*__sv__*.json' | wc -l       # 5
```

## 10. Generate the final tables and plots

Once all TEST evaluations have finished:

```bash
sbatch jobs/run_results.sh --target both --split test
```

- `--target both` includes DE and SV
- `--split test` reads TEST metrics and predictions; it **does not perform inference**

Look in `outputs/report/` for the combined Micro F1 chart, DEV prompt-sensitivity chart, per-emotion heatmap, main comparison table, language-transfer table, error examples, output-behavior table and runtime CSV. The exact filenames are listed in [README.md](README.md).

## 11. Check jobs and troubleshoot

```bash
squeue -u "$USER"
sacct -j <JOBID> --format=JobID,State,ExitCode
```

- `squeue` shows queued/running jobs; `sacct` shows whether a finished job succeeded or failed
- Replace `<JOBID>` with the ID printed by `sbatch`

Common problems:

- **No EuroLLM output:** confirm `role: secondary` and `--models secondary`
- **Missing five-shot IDs:** populate `prompting.five_shot_ids.de` and `.sv`; don't reuse one language's IDs for the other
- **Missing TEST selection:** run DEV selection first and check the language-specific JSON file
- **Incomplete TEST results:** wait for all 8 prompting and 5 PEFT results per language
- **Home directory full:** use scratch for the uv installation cache, and confirm `HF_HOME` is set in the Slurm job
- **Incorrect language:** ensure the Slurm script forwards `"$@"` so `--language` reaches `run_prompting.py`
