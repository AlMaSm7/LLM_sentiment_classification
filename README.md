# Cross-lingual multi-label emotion classification

This project investigates whether **parameter-efficient fine-tuning (PEFT)** improves emotion classification compared with **zero-shot and five-shot prompting**, and how much the wording of a prompt affects performance.

We study two language pairs separately: **English–German (DE)** and **English–Swedish (SV)**, using XED-based data. A text may contain more than one of the eight emotions: `anger`, `anticipation`, `disgust`, `fear`, `joy`, `sadness`, `surprise`, and `trust`.

## Models and experiments

| Model | Role | Methods |
|---|---|---|
| Qwen2.5-3B-Instruct | `primary` | PEFT, zero-shot, five-shot |
| EuroLLM-9B-Instruct | `secondary` | Zero-shot, five-shot |

PEFT has three training conditions: **English-only**, **target-language-only**, and **bilingual**. Prompting compares four instruction versions: `default`, `paraphrase`, `no_format_example`, and `terse`.

We use **DEV** to compare prompt wordings and select the best variant. We evaluate the selected variants on **TEST** without further tuning. Our main performance measure is **Micro F1**, supported by Macro F1, sample Jaccard, per-emotion F1, and malformed-output rates.

## Project structure

```text
configs/
  data.yaml                  # dataset preparation settings
  peft.yaml                  # fine-tuning settings
  prompting.yaml             # models, decoding and five-shot example IDs

scripts/                     # command-line entry points
  run_data_preparation.py
  run_peft_training.py
  run_peft_evaluation.py
  run_prompting.py
  run_results.py

src/llm_project/             # implementation
  data_preparation/
  peft_training.py
  peft_evaluation.py
  prompting.py
  results.py
  metrics.py                  # shared parser and metrics
  notebook_utils.py
  constants.py
  utils.py

jobs/                        # Slurm scripts for Hábrók
  run_data_preparation.sh
  run_peft_training.sh
  run_peft_evaluation.sh
  run_prompting.sh
  run_results.sh

outputs/                     # generated models, predictions, scores and figures
.env.example                 # optional local environment template
README.md
RUNNING_GUIDE.md
pyproject.toml
uv.lock
```

Files in `scripts/` accept command-line arguments. The actual processing is implemented in `src/llm_project/`. Prompting and PEFT share the same output parser and scoring functions in `metrics.py`.

## Workflow

1. Prepare the English–German and English–Swedish datasets.
2. Train Qwen PEFT adapters for the three training conditions in each language pair.
3. Evaluate PEFT on DEV and compare zero-shot/five-shot prompting across four prompt versions on DEV.
4. Select the best prompt version for **each target language**, using DEV results only.
5. Run the final prompting and PEFT evaluations on TEST.
6. Generate combined German/Swedish result tables and figures.

For each target language, prompting produces **32 DEV experiments** and **8 selected TEST experiments**. PEFT contributes **5 TEST evaluations** per target. The aligned English examples may differ between German and Swedish, so we treat these as separate evaluation cohorts.

## Before running

- Install the project environment and check your Hábrók Slurm scripts as explained in [RUNNING_GUIDE.md](RUNNING_GUIDE.md).
- Check `configs/data.yaml` and `configs/peft.yaml` before preparing data or training.
- Check the model names and roles in `configs/prompting.yaml`. Use `primary` and `secondary`.
- Store **five aligned demonstration IDs for each target language** under `prompting.five_shot_ids.de` and `prompting.five_shot_ids.sv`. The running guide explains how to find them.
- Select the prompting language with **`--language de` or `--language sv`**. You no longer need to change `language:` in the YAML file between prompting jobs.
- `.env.example` is a template for optional local settings. Copy it to `.env` only if needed, fill in your values, and never commit secrets. A `.env` file is not automatically loaded by Python or Slurm.

**All setup instructions and exact commands are in [RUNNING_GUIDE.md](RUNNING_GUIDE.md).**

## Expected results

The analysis reads saved predictions and metrics and writes report-ready files to `outputs/report/`. The main outputs are:

| Output | What it shows |
|---|---|
| `combined_test_main_table.tex` / `combined_test_main_results.csv` | PEFT vs zero-shot vs five-shot results for DE and SV |
| `combined_test_method_comparison.png` | Main Micro F1 comparison |
| `combined_dev_prompt_sensitivity.png` | Effect of the four prompt wordings on DEV |
| `combined_test_per_emotion_f1.png` | F1 score for each emotion |
| `combined_test_language_transfer_table.tex` | English vs German/Swedish evaluation differences |
| `combined_test_qualitative_candidates.csv` | Examples of correct and incorrect predictions |
| `combined_test_output_behavior_table.tex` | Malformed and empty model outputs |
| `combined_test_runtime.csv` | Recorded runtime information |

DEV prompt selections are kept separately in `selected_prompt_variants_de.json` and `selected_prompt_variants_sv.json`. The final analysis does not rerun model inference.
