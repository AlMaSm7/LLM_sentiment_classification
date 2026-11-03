# Cross-lingual multi-label emotion classification

This project investigates **whether parameter-efficient fine-tuning (PEFT) improves multi-label emotion classification compared with zero-shot and five-shot prompting**, how sensitive prompting is to instruction wording, and how well the methods transfer between **English and Swedish**. **Swedish is the only target language.**

The data comes from XED-based, aligned English–Swedish subtitle sentences. Each sentence may have more than one emotion: `anger`, `anticipation`, `disgust`, `fear`, `joy`, `sadness`, `surprise`, or `trust`. Neutral-only examples are excluded, so every retained reference has at least one emotion. An empty prediction (`[]`) is considered malformed and receives no predicted labels for scoring.

## Models and experiments

| Model | Role | Methods |
|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct` | `primary` | QLoRA PEFT, zero-shot, five-shot |
| `utter-project/EuroLLM-9B-Instruct` | `secondary` | Zero-shot, five-shot |

Qwen is fine-tuned in three conditions: **English-only**, **Swedish-only**, and **bilingual (English + Swedish)**. The evaluation matrix includes English and Swedish for English-only and bilingual PEFT, and Swedish for Swedish-only PEFT: **five PEFT evaluations per split**.

Prompting uses four instruction variants: `default`, `paraphrase`, `no_format_example`, and `terse`. Both models are evaluated on the English and Swedish texts using zero-shot and five-shot prompting.

| Stage | Experiments |
|---|---:|
| Prompting DEV | 2 models × 2 languages × 2 shot conditions × 4 variants = **32** |
| Prompting TEST | 2 models × 2 languages × 2 shot conditions × 1 selected variant = **8** |
| PEFT DEV / TEST | English-only (EN + SV), Swedish-only (SV), bilingual (EN + SV) = **5 per split** |

**Micro-F1** is the primary metric. Supporting analyses include sample Jaccard, paired bootstrap confidence intervals, output-format reliability, and average predicted labels per sentence.

## Project structure

```text
configs/
  data.yaml
  peft.yaml
  prompting.yaml

scripts/
  run_data_preparation.py
  run_peft_training.py
  run_peft_evaluation.py
  run_prompting.py

src/llm_project/
  data_preparation/
  peft_training.py
  peft_evaluation.py
  prompting.py
  metrics.py
  notebook_utils.py
  constants.py
  utils.py
  wandb_logging.py

jobs/
  run_data_preparation.sh
  run_peft_training.sh
  run_peft_evaluation.sh
  run_prompting.sh

notebooks/
  generate_report_results.ipynb

outputs/
  models/
  predictions/
  metrics/
  report/
  jobs/

.env.example
README.md
RUNNING_GUIDE.md
pyproject.toml
uv.lock
```

The `scripts/` files are the command-line entry points, while `src/llm_project/` contains the implementation. The **notebook** reads saved prediction and metric files; it does not train or run a language model.

## Workflow

1. Prepare the aligned English–Swedish TRAIN, DEV, and TEST data
2. Train the three Qwen PEFT adapters and evaluate them on DEV
3. Select five aligned English–Swedish training examples and run all **32 prompting DEV experiments**
4. Run **sections 1–3** of `notebooks/generate_report_results.ipynb` to choose prompt variants from DEV and save `outputs/report/selected_prompt_variants_sv.json`
5. Run the **8 selected prompting TEST experiments** and **5 PEFT TEST evaluations**
6. Run the **entire notebook** to display the results and save the Swedish TEST figure

For exact Hábrók commands and setup instructions, see [RUNNING_GUIDE.md](RUNNING_GUIDE.md).

## Results notebook

The notebook selects **one prompt wording per model and shot condition** using mean DEV Micro-F1 across English and Swedish, with mean sample Jaccard as the tie-breaker. The selected wording is reused for both input languages. The JSON contains eight model/language/shot keys and must exist **before TEST prompting**.

After TEST, the notebook creates these pandas DataFrames:

| Variable | Contents |
|---|---|
| `table2` | Qwen TEST Micro-F1, with DEV Micro-F1 in parentheses |
| `table3` | Qwen TEST empty-answer rates and average predicted labels per text |
| `table5` | DEV malformed-output and strict-JSON rates for both prompting models |
| `table4` | Short paired bootstrap summary for Micro-F1 |
| `table7` | Complete paired bootstrap comparisons for Micro-F1 |
| `table4_jaccard` | Short paired bootstrap summary for sample Jaccard |

The bootstrap uses **1,000 paired resamples** and a fixed seed of **42**. Comparisons require matching alignment IDs and gold labels; partial-overlap comparisons may be marked `†`, while insufficiently paired comparisons appear as `—`.

The notebook saves only these report artifacts:

```text
outputs/report/selected_prompt_variants_sv.json
outputs/report/sv_test_method_comparison.png
```


## Before running

- Verify that `configs/data.yaml` targets Swedish and `data/processed/sv/` contains aligned English and Swedish data
- Configure `primary` and `secondary` model roles and five distinct aligned TRAIN demonstration IDs in `configs/prompting.yaml`
- Verify the five-shot selector checks **English–Swedish only**, not a former German–Swedish setup
- Confirm the Slurm PEFT array indices corresponding to the three Swedish training conditions
- Use `--language sv` for prompting and complete all DEV runs before selecting the prompt variants
- Do not select prompts, checkpoints, or hyperparameters using TEST results
- Store Hugging Face and uv caches under scratch, as described in the running guide
- Keep `.env` files and access tokens out of Git; `.env` is not loaded automatically by Python or Slurm
