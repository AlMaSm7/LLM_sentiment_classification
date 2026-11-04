# Running guide — English–Swedish emotion classification

This guide covers the **Swedish target-language experiment** on the University of Groningen Hábrók cluster. Run commands from the repository root. The project uses English–Swedish aligned XED/OPUS subtitles, Qwen2.5-3B-Instruct (prompting and QLoRA), and EuroLLM-9B-Instruct (prompting only).

## 1. Set up the environment

```bash
cd /scratch/$USER/llm/LLM_sentiment_classification
module load Python/3.12.3-GCCcore-13.3.0
mkdir -p outputs/jobs jobs/logs

export HF_HOME="/scratch/$USER/llm/hf_cache"
export UV_CACHE_DIR="/scratch/$USER/llm/uv_cache"
export TMPDIR="/scratch/$USER/llm/tmp"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"

uv sync --locked --no-dev
uv run --locked --no-dev --no-sync python -m compileall -q src scripts
```

The Slurm scripts must export the same cache paths. `uv run --locked --no-dev --no-sync` reuses the prepared environment without installing dependencies in a job. Never commit `.env` files or access tokens.

**Weights & Biases (W&B):** Optional experiment tracking for PEFT training. Set `WANDB_API_KEY` in `.env` and enable it in `configs/peft.yaml`:

- `use_wandb: true` — enable logging
- `use_wandb: false` — disable logging
- `wandb_mode: "offline"` — save logs locally without internet access

## 2. Data-preparation stages

`configs/data.yaml` should identify the Swedish target and fixed split seed (currently `40`):

```yaml
source_language: en
target_language: sv

opus:
  corpus: OpenSubtitles
  release: v2018

split:
  seed: 40
  train: 0.7
  dev: 0.2
  test: 0.1
```

`run_data_preparation.py` requires `--stage`:

| Flag | Effect | Changes existing splits? | Downloads? |
|---|---|---|---|
| `--stage download` | Fetch missing XED projections and export OPUS EN–SV pairs/IDs | No | Yes, if needed |
| `--stage split` | Match XED to cached OPUS, clean, stratify, and save JSONL files | Creates initial split; checks against existing assignments | No |
| `--stage rebalance` | Rebuild **only** `train_bilingual.jsonl` from existing `train_en.jsonl` and `train_sv.jsonl` | **No** | **No** |
| `--stage all` | Download and then create the split | Creates/checks initial split | If needed |

### First-time setup: download, then split

Submit these jobs **one after the other** (wait for the download job to finish before starting the split job):

```bash
sbatch jobs/run_data_preparation.sh --stage download
# wait until the download has completed successfully
sbatch jobs/run_data_preparation.sh --stage split
```

The download stage creates/caches these files:

```text
data/raw/xed/sv-projections.tsv
data/raw/opus/en-sv/en-sv.en.txt
data/raw/opus/en-sv/en-sv.sv.txt
data/raw/opus/en-sv/en-sv.ids.tsv
data/raw/opus/cache/...
```

The split stage uses these local files only. It saves `train_en.jsonl`, `train_sv.jsonl`, `train_bilingual.jsonl` and the corresponding DEV and TEST files under `data/processed/sv/`.

It also saves `split_assignments.csv` and `split_manifest.json`. If existing split assignments would change, the code **raises an error rather than overwriting them**. Keep the original split unchanged throughout the experiment.

### Existing data: rebalance the bilingual TRAIN file only

This is the command to use **now** if you already have correct TRAIN/DEV/TEST files:

```bash
sbatch jobs/run_data_preparation.sh --stage rebalance
```

Or run it on a login/CPU environment directly:

```bash
uv run --locked --no-dev --no-sync python scripts/run_data_preparation.py --stage rebalance
```

The rebalance stage reads the existing `train_en.jsonl` and `train_sv.jsonl`, matches rows by `alignment_id`, verifies their gold emotion labels, and selects **one language per aligned pair**. It uses the fixed seed in `configs/data.yaml` (`split.seed`), so the output is reproducible. No XED or OPUS data is downloaded, and **all English-only, Swedish-only, DEV and TEST files stay unchanged**.

For example, if TRAIN contains 1,733 aligned pairs, expected bilingual TRAIN is 867 English + 866 Swedish = 1,733 total examples. No two bilingual examples share an `alignment_id`.

Check the generated counts:

```bash
wc -l data/processed/sv/train_*.jsonl
```

Verify the bilingual language mix and ID uniqueness:

```bash
uv run --locked --no-dev --no-sync python - <<'PY'
import json
from collections import Counter
from pathlib import Path

folder = Path("data/processed/sv")

def rows(filename):
    with (folder / filename).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]

en = rows("train_en.jsonl")
sv = rows("train_sv.jsonl")
bi = rows("train_bilingual.jsonl")
ids = [str(row["alignment_id"]) for row in bi]

print("English-only:", len(en))
print("Swedish-only:", len(sv))
print("Bilingual:", len(bi), Counter(row["language"] for row in bi))
assert len(en) == len(sv) == len(bi)
assert len(ids) == len(set(ids))
print("Balanced training validation passed")
PY
```

**Note:** The data preparation seed (`configs/data.yaml` → `split.seed`) determines the language choice within the bilingual dataset. The separate PEFT seed (`configs/peft.yaml` → `experiment.seed`) controls training randomness.

**After rebalancing, retrain the bilingual adapter** because its training data has changed. English-only and Swedish-only adapters do not need retraining if their original files and settings are unchanged. Keep DEV and TEST evaluation data identical to previous runs; replace or clearly version stale bilingual evaluation results.

## 3. Verify model cache

The models are:

```text
Qwen/Qwen2.5-3B-Instruct
utter-project/EuroLLM-9B-Instruct
```

If not already downloaded, cache them on a network-enabled node:

```bash
unset HF_HUB_OFFLINE
uv run --locked --no-dev --no-sync python - <<'PY'
from huggingface_hub import snapshot_download
for model in ("Qwen/Qwen2.5-3B-Instruct", "utter-project/EuroLLM-9B-Instruct"):
    print(model, snapshot_download(repo_id=model))
PY
```

List cached model repositories:

```bash
find "${HF_HUB_CACHE:-$HF_HOME/hub}" -maxdepth 1 -type d -name 'models--*' \
    -printf '%f\n' | sed 's/^models--//; s/--/\//g' | sort
```

A listed repository name does **not** guarantee a complete model download. Set `HF_HUB_OFFLINE=1` in GPU jobs only after the model and tokenizer load successfully from cache.

## 4. Train and evaluate QLoRA

The three PEFT conditions are English-only, Swedish-only (`target`), and bilingual. The training script reads the prepared files; **it should not perform balancing**. Remove any old balancing block from `scripts/run_peft_training.py`.

Train using your current no-array Slurm job:

```bash
sbatch jobs/run_peft_training.sh
```

After training, evaluate the adapters on DEV:

```bash
sbatch jobs/run_peft_evaluation.sh dev
```

Your current training job should iterate over the three conditions or call the entry point with separate `--condition` arguments; **do not introduce old Slurm array-index mappings**. If you only need to retrain bilingual, use a single-condition execution supported by your training job or call the Python entry point with `--language sv --condition bilingual` within an allocated GPU job.

There should be five PEFT evaluations per split: EN-only on EN/SV, SV-only on SV, and bilingual on EN/SV. W&B is optional according to `use_wandb` in `configs/peft.yaml`; it does not affect dataset preparation.

## 5. Select five-shot demonstrations

After TRAIN data is available:

```bash
uv run --locked --no-dev --no-sync python scripts/run_prompting.py --select-ids
```

`prompting.py` now selects five aligned English–Swedish examples from TRAIN only. Enter those five IDs in `prompting.five_shot_ids` in `configs/prompting.yaml`. The IDs are shared across the English and Swedish versions. Rebalancing `train_bilingual.jsonl` does **not** change the English/Swedish TRAIN files or these demonstrations.

## 6. Run prompting on DEV

```bash
sbatch jobs/run_prompting.sh dev --sensitivity --models primary
sbatch jobs/run_prompting.sh dev --sensitivity --models secondary
```

DEV prompting runs: 2 models × 2 input languages × zero/five-shot × 4 prompt variants = **32 configurations**. Use only DEV to choose the wording. The four variants are `default`, `paraphrase`, `no_format_example`, and `terse`.

Open `notebooks/generate_report_tables_simple.ipynb` and run its DEV selection cells. The selected wordings are saved to:

```text
outputs/report/selected_prompt_variants_sv.json
```

## 7. Final TEST evaluation

After the DEV-selected prompting variants are saved:

```bash
sbatch jobs/run_prompting.sh test --models primary
sbatch jobs/run_prompting.sh test --models secondary
sbatch jobs/run_peft_evaluation.sh test
```

Expected: **8 prompting TEST** runs and **5 PEFT TEST** evaluations. Do not use TEST to choose the prompt wording, demonstrations, checkpoint, or training balance.

## 8. Report and checks

Run `notebooks/generate_report_tables_simple.ipynb` after all evaluations. It reads saved metrics and predictions, produces the report tables, and generates `outputs/report/sv_test_method_comparison.png`. It does not run model inference.

Use `notebooks/data_check.ipynb` to inspect the label distributions of TRAIN, DEV and TEST and the language composition of the three PEFT training datasets.

Cluster monitoring:

```bash
squeue -u "$USER"
```

```bash
sacct -j <JOBID> --format=JobID,State,ExitCode
```

If any code in your local repository still has old job names, or Slurm arrays, update it to match the current Swedish-only workflow before running the experiment.
