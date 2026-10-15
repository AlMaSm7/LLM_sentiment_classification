# Running PEFT and Prompting Experiments

This guide explains how to run the main training and inference workflows in this project.

## 1. Environment setup

From the project root:

```bash
cd /path/to/LLM_sentiment_classification
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

If you use `uv` instead:

```bash
cd /path/to/LLM_sentiment_classification
uv sync
source .venv/bin/activate
```

Make sure the project source is on `PYTHONPATH` when running scripts directly:

```bash
export PYTHONPATH=$PWD/src
```
Login to hf
```
hf auth login

## 2. Prepare the data

Before running experiments, prepare the processed data:

```bash
python scripts/prepare_data.py
```

This uses `configs/data.yaml` and writes output into `data/processed/`.

## 3. PEFT workflow

The PEFT configuration is in `configs/peft.yaml`.

Example training run:

```bash
export PYTHONPATH=$PWD/src
python scripts/run_peft.py --language sv --condition target --split dev

or

sbatch --array=0-2 jobs/peft_jobs.sh     # all three conditions
sbatch --array=1 jobs/peft_jobs.sh       # only the target-language condition
```

This script will:

1. load model/training config
2. load training data
3. build the base model and tokenizer
4. apply LoRA/QLoRA
5. train the adapter
6. save the model under `outputs/models/`
7. evaluate on the selected split
8. run a few inference examples

You can also evaluate an already-trained adapter without retraining:

```bash
python scripts/evaluate_peft.py --language sv --condition target --split dev

# re-score the PEFT predictions with the same parser and metrics as prompting
```
python scripts/rescore_predictions.py --inputs "outputs/predictions/peft_raw/dev/*.jsonl" --family peft --split dev
```

## 4. Prompting workflow

The prompting configuration is in `configs/prompting.yaml`.

Before running five-shot prompting, select demonstration examples once:

```bash
python scripts/run_prompting.py --select-ids
```

This prints five aligned examples and shows the JSON snippet to paste into:

```yaml
prompting:
  five_shot_ids: [ ... ]
```

Then update `configs/prompting.yaml` with those IDs.

Run prompting on the dev split:

```bash
python scripts/run_prompting.py --split dev
sbatch jobs/run_prompting.sh dev
```

Run prompting with wording sensitivity analysis (dev only):

```bash
python scripts/run_prompting.py --split dev --sensitivity
sbatch jobs/run_prompting.sh dev --sensitivity

On both models
```bash
sbatch jobs/run_prompting.sh dev --sensitivity wording --variants paraphrase no_format_example terse --models primary
sbatch jobs/run_prompting.sh dev --sensitivity wording --variants paraphrase no_format_example terse --models second
```

Follow the progrress of prompting jobs
````
grep macro_f1 $(ls -t jobs/logs/prompting_*.out | head -n 1)
```

Run prompting on the test split:

```bash
python scripts/run_prompting.py --split test
```

### Quick smoke test

```bash
python scripts/run_prompting.py --split dev --limit 20
```

Restrict to one model role:

```bash
python scripts/run_prompting.py --split dev --models primary
```

## 5. Run both together

There is a convenience wrapper script:

```bash
chmod +x scripts/compare_peft_prompting.sh
./scripts/compare_peft_prompting.sh --dry-run --all
```

To actually execute it:

```bash
./scripts/compare_peft_prompting.sh --run --all
```

or
```
sbatch jobs/run_prompting.sh dev
sbatch --array=0-2 jobs/peft_jobs.sh
```

Available options:

- `--train`
- `--peft-infer`
- `--prompt`
- `--all`
- `--dry-run`
- `--run`

## 6. Output folders

PEFT outputs:

```text
outputs/models/
outputs/predictions/
outputs/metrics/
```

Prompting outputs:

```text
outputs/predictions/prompting/
outputs/metrics/prompting/
```

## 7. Typical quick start

```bash
cd /scratch/$USER/LLM_sentiment_classification
source /scratch/$USER/.llm/bin/activate
export PYTHONPATH=$PWD/src

python scripts/prepare_data.py
python scripts/run_prompting.py --select-ids
# paste IDs into configs/prompting.yaml

sbatch --array=0-2 jobs/peft_jobs.sh
sbatch jobs/run_prompting.sh dev

#
--select-ids prints the selected demonstrations;
--sensitivity runs the extra wordings (dev only; it refuses to run on test);
--list-variants checks every wording against the spec;
--skip-existing resumes after a timeout;
smoke tests (--limit) write to a separate folder.

# when the PEFT jobs have finished:
python scripts/rescore_predictions.py --inputs "outputs/predictions/peft_raw/dev/*.jsonl" --family peft --split dev
```

# check status of jobs
squeue -u $USER

## 8. Notes

- `configs/peft.yaml` configures LoRA/QLoRA training, model choice, learning rate, and batch size.
- `configs/prompting.yaml` configures model roles, language, and the five-shot prompt examples.
- W&B is disabled by default in the PEFT config, but can be enabled if desired.
- If you get import errors, make sure the venv is activated and `PYTHONPATH` is set.

