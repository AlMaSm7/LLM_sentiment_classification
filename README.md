# Cross-lingual multi-label emotion classification

A reproducible student project comparing zero-shot prompting, five-shot prompting, and parameter-efficient fine-tuning (PEFT) on matched English–target-language XED movie subtitles. The main YAML configurations currently select **Swedish (`sv`)** as the assigned language. The pipeline also supports German (`de`) as a separately configured experiment.

**Primary model:** `Qwen/Qwen2.5-3B-Instruct`  
**Second prompting model:** `utter-project/EuroLLM-9B-Instruct`  
**Labels:** anger, anticipation, disgust, fear, joy, sadness, surprise, trust

## Repository layout

```text
configs/                       data, prompting and PEFT settings
src/llm_project/
  constants.py                 shared label names and data-source URL
  data_preparation/            XED/OPUS matching and fixed stratified splits
  peft_training.py             LoRA/QLoRA model setup and fine-tuning
  peft_evaluation.py           PEFT inference and raw prediction export
  peft_inference.py            small inference examples
  prompting.py                 exact prompts, demonstrations and shared parser
  prompt_evaluation.py         single scoring implementation for both methods
scripts/
  prepare_data.py              data preparation entry point
  run_peft.py                  training and development evaluation
  evaluate_peft.py             reload and evaluate an existing adapter
  rescore_predictions.py      apply shared parser and metrics to PEFT outputs
  run_prompting.py            zero-shot, five-shot and wording variants
  analyze_results.py          comparison tables, plots and qualitative analysis
jobs/                          CPU and GPU Slurm scripts
notebooks/                     optional exploratory analysis, with outputs removed
report/report.tex              Overleaf-ready technical report draft
README.md                      project overview
RUNNING_GUIDE.md               full command sequence
pyproject.toml / uv.lock        environment and pinned dependencies
```

Generated datasets, model weights, predictions, caches, Slurm logs, and credentials are **not committed**. The uploaded archive contained a `.git` directory and `.env`; neither is included in the reviewed deliverable.

## Reproduction in brief

Run commands from the repository root:

```bash
module load Python/3.12.3-GCCcore-13.3.0
uv sync --locked
mkdir -p outputs/jobs jobs/logs
sbatch jobs/prepare_data.sh
```

After data preparation has completed, run the appropriate training and evaluation jobs. See **[RUNNING_GUIDE.md](RUNNING_GUIDE.md)** for the full experiment matrix, prompt selection, evaluation, and report generation. Check the cluster's internet policy before downloading XED/OPUS resources from compute nodes.

## Data and experimental design

The pipeline uses XED projected annotations and matched English–target-language OpenSubtitles v2018 pairs. The target-language XED labels are projected onto subtitles; the English side inherits the same labels for matched pairs. The data preparation code uses Unicode NFKC and whitespace normalization, removes conflicting and ambiguous matches, and creates a fixed **70/20/10 train/DEV/TEST split** using multi-label stratification (seed **40**). Training and five-shot selection use seed **42**.

**Important provenance caveat:** the current pipeline constructs identifiers from OPUS document and sentence IDs; the brief requires original XED alignment IDs for the split. These are not proven equivalent. Verify the original source-ID relationship before claiming strict compliance with that requirement. Do not use the held-out TEST data to choose prompts or hyperparameters.

PEFT uses three training conditions: English-only, target-language-only, and bilingual. The required evaluation matrix has five PEFT comparisons per target language: English-only on both languages, target-only on target, and bilingual on both. Prompting uses Qwen and EuroLLM with zero/five shots on both input languages. Four wording variants are compared on DEV, and one selected variant per model and prompting condition is frozen before TEST.

## Metrics and results

All **reported** metrics must come from the shared implementation in `src/llm_project/prompt_evaluation.py`. Both prompting and PEFT use the same strict JSON-list parser from `src/llm_project/prompting.py` and are scored on eight binary labels in fixed order. Invalid formats and unpermitted label names become malformed empty predictions; changing the parser requires rescoring existing raw outputs. Primary metrics: Micro F1 and **sample-averaged Jaccard**. Secondary: Macro F1, micro precision/recall, and Hamming loss. Diagnostic: per-emotion precision/recall/F1/support, exact match, label cardinality, empty predictions, and malformed-rate. Zero division is set to zero.

The standalone PEFT training job additionally prints its own legacy metrics; **do not mix its micro-averaged Jaccard with the shared sample Jaccard**. Run `scripts/rescore_predictions.py` and use only those common metrics for report comparisons.

`uv run python scripts/analyze_results.py --target sv --split test` produces comparison tables, per-emotion tables, prompt-sensitivity figures, test error examples, and Micro F1 plots in `outputs/report/`. The script also creates a runtime table where hardware metadata is available. It does not invent missing data; it requires the corresponding saved experiment outputs.

## Scientific and implementation caveats

- The original archive's `qlora` branch used **8-bit** loading despite an NF4 YAML configuration. The reviewed code switches that branch to **4-bit NF4**. Previously trained adapters do **not** become 4-bit QLoRA merely by updating code. Retrain if you want to claim the corrected setup
- Exact OPUS--XED ID provenance is still unverified
- The source ZIP did not contain the final metric artifacts, so the current report's numeric results remain provisional until checked
- Runtime and peak GPU memory should be recorded from new jobs or Slurm accounting rather than guessed
- Changing prompts, parsers, splits or quantization creates a new experimental condition; do not silently combine old and new scores

## Checks

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src scripts
uv run ruff check src scripts
uv run ruff format --check src scripts
```

Ruff formatting/lint may identify additional style issues in legacy modules. Syntax and small CPU-only tests have been verified for the reviewed code, but full GPU runs and source-data downloads require the Hábrók environment.

## References

- [XED dataset](https://github.com/Helsinki-NLP/XED)
- [OpusTools](https://github.com/Helsinki-NLP/OpusTools)
- [PEFT](https://huggingface.co/docs/peft)
- [Course report source](report/report.tex)
