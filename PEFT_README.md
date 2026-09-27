# PEFT Workflow

This folder provides a shared starting structure for PEFT experiments. The prepared datasets in `data/processed/` should be reused as-is so that all implementations work with the same train/dev/test splits.

## Data

Processed data is grouped by target language:

```text
data/processed/
├── de/
│   ├── train_en.jsonl
│   ├── train_de.jsonl
│   ├── train_bilingual.jsonl
│   ├── dev_en.jsonl
│   ├── dev_de.jsonl
│   ├── dev_bilingual.jsonl
│   ├── test_en.jsonl
│   ├── test_de.jsonl
│   └── test_bilingual.jsonl
│
└── sv/
    ├── train_en.jsonl
    ├── train_sv.jsonl
    ├── train_bilingual.jsonl
    ├── dev_en.jsonl
    ├── dev_sv.jsonl
    ├── dev_bilingual.jsonl
    ├── test_en.jsonl
    ├── test_sv.jsonl
    └── test_bilingual.jsonl
```

Each language folder is self-contained. For example, the English data inside `de/` is the English side of the English–German matched examples, while the English data inside `sv/` belongs to the English–Swedish matched examples.

The three PEFT training conditions are:

```text
english    -> train_en.jsonl
target     -> train_<language>.jsonl
bilingual  -> train_bilingual.jsonl
```

Use the corresponding development split while developing and selecting a setup. The test split is intended for final evaluation.

## Template files

```text
configs/
└── peft_template.yaml

src/llm_project/
├── peft_model.py
├── peft_training.py
└── peft_evaluation.py

scripts/
└── train_peft.py

jobs/
└── peft_template.slurm
```

The files only provide a common structure and a few placeholder functions. The actual model choice, LoRA/QLoRA setup, trainer, prompting format, and training strategy can be implemented independently.

## Configuration

Copy or adapt `configs/peft_template.yaml` for an experiment. It contains the main values that are useful to keep explicit, such as the model name, LoRA settings, training settings, random seed, and output directory.

Example:

```yaml
model:
  name: "your-model"

peft:
  method: "lora"
  rank: 16
  alpha: 32
  dropout: 0.05

training:
  learning_rate: 0.0002
  batch_size: 4
  epochs: 3
```

## Running an experiment

The shared training script accepts:

```text
--config
--language
--condition
```

Example for English-only training on the English–German dataset:

```bash
uv run python scripts/train_peft.py \
    --config configs/peft_template.yaml \
    --language de \
    --condition english
```

Example for bilingual training on the English–Swedish dataset:

```bash
uv run python scripts/train_peft.py \
    --config configs/peft_template.yaml \
    --language sv \
    --condition bilingual
```

The script resolves the matching files automatically. For example:

```text
--language de --condition target
```

uses:

```text
data/processed/de/train_de.jsonl
data/processed/de/dev_de.jsonl
data/processed/de/test_de.jsonl
```

The same structure works for Swedish by replacing `de` with `sv`.

## Evaluation

Predictions should use the eight XED emotion labels:

```text
anger
anticipation
disgust
fear
joy
sadness
surprise
trust
```

The shared evaluation module contains example helpers for parsing predictions and computing common multi-label metrics. Implementations can extend these helpers if needed, but results should be reported consistently across the PEFT conditions.

## Outputs

```text
outputs/
├── models/<run_name>/       # adapter, tokenizer, and Trainer checkpoints
├── metrics/<run_name>/      # metrics.json
└── predictions/<run_name>/  # predictions.jsonl and predictions.csv
```

The run name is taken from `--run-name`, then `WANDB_NAME`, or generated from
the PEFT method, language, and condition. The same run name is used for the
local output directories and the W&B run.

Large checkpoints and experiment outputs should normally stay out of Git.
