# Running guide — English–Swedish

This guide explains how to run the **Swedish-only target experiment** on the University of Groningen **Hábrók HPC cluster**, then prepare the report results in Jupyter. Run shell commands from the repository root unless specified otherwise.

## 1. Prepare the Python environment

Connect to Hábrók and open the repository:

```bash
cd /scratch/$USER/llm/LLM_sentiment_classification
module load Python/3.12.3-GCCcore-13.3.0
mkdir -p outputs/jobs jobs/logs /scratch/$USER/llm/uv_cache /scratch/$USER/llm/tmp
```

Install the locked environment once:

```bash
UV_CACHE_DIR="/scratch/$USER/llm/uv_cache" \
TMPDIR="/scratch/$USER/llm/tmp" \
uv sync --locked --no-dev
```

For commands run **manually** rather than through Slurm, set the cache locations in your terminal:

```bash
export HF_HOME="/scratch/$USER/llm/hf_cache"
export UV_CACHE_DIR="/scratch/$USER/llm/uv_cache"
export TMPDIR="/scratch/$USER/llm/tmp"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"
```

The Slurm job scripts should configure their own cache locations and load Python. Confirm they use scratch rather than the limited home directory. An optional `.env` file can be created from `.env.example`, but it is **not** automatically loaded by Slurm or Python.

Log in to wandb, if `use_wandb: true` in `configs/peft.yaml`:

```bash
uv run --locked --no-dev --no-sync wandb login
```

Or use set the `WANDB_API_KEY` in `.env`.

## 2. Cache both models

Download the model files once on a node where Hugging Face downloads are **permitted**:

```bash
unset HF_HUB_OFFLINE
uv run --locked --no-dev --no-sync python - <<'PYCODE'
from huggingface_hub import snapshot_download

for model in (
    "Qwen/Qwen2.5-3B-Instruct",
    "utter-project/EuroLLM-9B-Instruct",
):
    print(model, snapshot_download(repo_id=model))
PYCODE
```

Existing cached files are reused. If authentication is needed, log in with a read-access token without storing it in the repository. GPU jobs can use `HF_HUB_OFFLINE=1` after both models are fully cached.

Check cached Hugging Face Models:

```bash
export HF_HOME="/scratch/$USER/llm/hf_cache"

find "${HF_HUB_CACHE:-$HF_HOME/hub}" \
    -maxdepth 1 \
    -type d \
    -name "models--*" \
    -printf '%f\n' \
    | sed 's/^models--//; s/--/\//g' \
    | sort
```

Expected output:

```bash
Qwen/Qwen2.5-3B-Instruct
utter-project/EuroLLM-9B-Instruct
```

## 3. Confirm the Swedish configuration and job scripts

Set the dataset target to Swedish in `configs/data.yaml`. In `configs/prompting.yaml`, use these model roles:

```yaml
models:
  - name: Qwen/Qwen2.5-3B-Instruct
    role: primary
  - name: utter-project/EuroLLM-9B-Instruct
    role: secondary
```

The five-shot prompting configuration uses a **single list of five distinct aligned IDs**:

```yaml
prompting:
  id_key: alignment_id
  five_shot_ids: []
  selection:
    k: 5
    seed: 42
    min_words: 5
    max_words: 20
    max_labels: 3
    min_single: 1
```

The intended Slurm/Python pairs are:

| Slurm job | Python entry point |
|---|---|
| `jobs/run_data_preparation.sh` | `scripts/run_data_preparation.py` |
| `jobs/run_peft_training.sh` | `scripts/run_peft_training.py` |
| `jobs/run_peft_evaluation.sh` | `scripts/run_peft_evaluation.py` |
| `jobs/run_prompting.sh` | `scripts/run_prompting.py` |

Check that `jobs/run_prompting.sh` forwards `"$@"` and accepts `dev` or `test` as its first positional argument. The final tables and figure are generated in Jupyter, not by a results Slurm job.

## 4. Prepare the English–Swedish datasets
Select correct target language and corpus in `configs/data.yaml`:
```yaml
source_language: en
target_language: sv

opus:
  corpus: OpenSubtitles
  release: v2018
```

```bash
sbatch jobs/run_data_preparation.sh
```

After the job finishes, check the processed folder:

```bash
ls data/processed/sv/
```

It should contain aligned English and Swedish TRAIN, DEV, and TEST records, including `train_en.jsonl`, `train_sv.jsonl`, `dev_en.jsonl`, `dev_sv.jsonl`, `test_en.jsonl`, and `test_sv.jsonl`. Each aligned pair shares an alignment ID and emotion labels. Neutral-only records are excluded. **Keep the split fixed** for the entire experiment.

## 5. Train and evaluate Qwen PEFT on DEV

Three adapters are required: **English-only**, **Swedish-only**, and **bilingual

```bash
sbatch jobs/run_peft_training.sh
```

**After training completes**, evaluate on DEV using the same indices:

```bash

sbatch jobs/run_peft_evaluation.sh dev
```

The expected evaluation matrix is **five PEFT results per split**: English-only on EN/SV, Swedish-only on SV, and bilingual on EN/SV. Results are stored under `outputs/metrics/peft/dev/` and `outputs/predictions/peft/dev/`.

## 6. Select the five-shot examples

Once English–Swedish TRAIN data is ready, select demonstrations:

```bash
uv run --locked --no-dev --no-sync python scripts/run_prompting.py --language sv --select-ids
```

The selected alignment IDs must refer to the **same labeled pair** in `data/processed/sv/train_en.jsonl` and `data/processed/sv/train_sv.jsonl`. Copy the five printed IDs into `prompting.five_shot_ids` in `configs/prompting.yaml`, preserving their numeric/string types and order.

## 7. Run prompting on DEV

```bash
sbatch jobs/run_prompting.sh dev --language sv --sensitivity --models primary
sbatch jobs/run_prompting.sh dev --language sv --sensitivity --models secondary
```

The full DEV matrix contains **32 prompting runs**: two models × English/Swedish inputs × zero/five-shot × four prompt variants. The variants are `default`, `paraphrase`, `no_format_example`, and `terse`.

Check the metric count after both jobs finish:

```bash
find outputs/metrics/prompting/dev -maxdepth 1 -name '*__sv__*.json' | wc -l
```

Expected: **32**.

When prompts, demonstration IDs, generation settings, or parser rules have changed, do not reuse outdated evaluations through `--skip-existing`. Neutral is not part of the label space; an output of `[]` is malformed and scored as an empty prediction.

## 8. Select the DEV prompting variants in Jupyter

Open:

```text
notebooks/generate_report_results.ipynb
```

Use the project's Python environment as the Jupyter kernel, for example via VS Code Remote SSH with the repository `.venv`. Run **Sections 1–3** of the notebook **after all 32 prompting DEV runs are complete**.

The notebook calculates mean DEV **Micro-F1** across English and Swedish for each prompt wording, breaks ties using mean sample Jaccard, and chooses **one variant per model and shot condition**. It stores the choice for both input languages in:

```text
outputs/report/selected_prompt_variants_sv.json
```

There are eight keys: Qwen/EuroLLM × English/Swedish × zero/five-shot, with the same chosen wording for both languages of a given model and shot condition. **TEST scores are not used to select the wording.**

The notebook checks whether a previously saved selection would change after TEST prompting has been generated. Resolve any inconsistency before replacing results.

## 9. Run the final TEST evaluations

Only begin after the selection JSON exists. Do not use `--sensitivity` or `--limit` on TEST.

Prompting TEST:

```bash
sbatch jobs/run_prompting.sh test --language sv --models primary
sbatch jobs/run_prompting.sh test --language sv --models secondary
```

PEFT TEST, after the trained adapters are available:

```bash
sbatch jobs/run_peft_evaluation.sh test
```

Once all evaluations finish, check the output counts:

```bash
find outputs/metrics/prompting/test -maxdepth 1 -name '*__sv__*.json' | wc -l  # 8
find outputs/metrics/peft/test -maxdepth 1 -name '*__sv__*.json' | wc -l       # 5
```

The prompting jobs read `selected_prompt_variants_sv.json` and evaluate only the DEV-selected wording for each model, language, and shot condition.

## 10. Create the report tables and figure

Open `notebooks/generate_report_tables_simple.ipynb` and **run all cells from top to bottom**. The notebook reads the saved metrics JSON and prediction JSONL files; it does not load any model or perform inference.

It displays these **pandas DataFrames**:

| Variable | Contents |
|---|---|
| `table2` | Qwen TEST Micro-F1 with DEV in parentheses |
| `table3` | Qwen TEST empty-answer rates and average predicted label counts |
| `table5` | DEV malformed-output and strict-JSON rates for Qwen and EuroLLM |
| `table4` | Short paired Micro-F1 bootstrap summary |
| `table7` | Full paired Micro-F1 bootstrap comparisons |
| `table4_jaccard` | Short paired sample-Jaccard bootstrap summary |

The notebook also saves:

```text
outputs/report/sv_test_method_comparison.png
```

The figure shows actual **saved TEST Micro-F1** scores for Qwen prompting, Qwen PEFT, and EuroLLM prompting.

Bootstrap results use **1,000 paired resamples** and seed 42. Each comparison matches predictions by `alignment_id` and verifies matching gold labels. If at least 80% of both runs overlap and at least 20 examples match, the comparison uses the shared examples; results with partial overlap are marked `†`. Otherwise the comparison is shown as `—`. A partial-overlap comparison describes only the shared subset, not necessarily the full TEST set.

A consistency-check cell compares saved Micro-F1 against scores reconstructed from the prediction files.

## 11. Monitor jobs and troubleshoot

```bash
squeue -u "$USER"
```

```bash
sacct -j <JOBID> --format=JobID,State,ExitCode
```

| Problem | What to check |
|---|---|
| German jobs are launched | Verify the PEFT array mapping and select only Swedish tasks |
| Five-shot selection requires German | Ensure the selector validates only the EN–SV training pair |
| Missing demonstrations | Check five distinct aligned TRAIN IDs and their gold labels |
| Missing EuroLLM runs | Confirm `role: secondary` and `--models secondary` |
| Missing selection JSON | Complete 32 prompting DEV runs, then run notebook Sections 1–3 |
| TEST prompting runs the wrong variant | Confirm the selection JSON exists and its keys match the prompting model names |
| Different saved and recalculated Micro-F1 | Investigate stale metrics, stale prediction files, or changed parsing rules |
| Bootstrap returns `—` or `†` | Check matching alignment IDs, example overlap, and gold labels |
| No report figure | Complete 8 prompting and 5 PEFT TEST evaluations, then run the notebook |
| Out of home disk space | Verify `HF_HOME`, `UV_CACHE_DIR`, and `TMPDIR` point to scratch |

For the overall project description and current reporting outputs, see [README.md](README.md).
