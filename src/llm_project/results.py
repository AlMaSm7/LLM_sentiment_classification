from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from llm_project.constants import EMOTIONS
from llm_project.metrics import gold_labels
from llm_project.utils import load_jsonl

RESULT_COLUMNS = (
    "run",
    "target_language",
    "split",
    "model",
    "family",
    "method",
    "training",
    "input_language",
    "variant",
    "n",
    "micro_f1",
    "sample_jaccard",
    "macro_f1",
    "micro_precision",
    "micro_recall",
    "hamming_loss",
    "exact_match",
    "malformed_rate",
    "empty_prediction_rate",
    "gold_label_cardinality",
    "pred_label_cardinality",
    "inference_seconds",
    "peak_gpu_memory_gb",
    "gpu",
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    # reuse the project jsonl loader for dataset and prediction files
    return cast(list[dict[str, Any]], load_jsonl(path).to_list())


def load_runs(project_root: Path, target: str, split: str) -> list[dict[str, Any]]:
    # load each saved experiment metric without running the model again
    runs = []
    for family in ("prompting", "peft"):
        folder = project_root / "outputs" / "metrics" / family / split
        for path in sorted(folder.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            parts = record.get("run", path.stem).split("__")
            if len(parts) != 5 or parts[1] != target:
                continue

            model, _, input_lang, condition, suffix = parts
            if family == "prompting":
                if condition not in ("zero_shot", "five_shot"):
                    continue
                method, training, variant = condition, "none", suffix
            else:
                if not condition.startswith("peft-"):
                    continue
                method, training, variant = "peft", condition.removeprefix("peft-"), ""

            if record.get("target_language", target) != target:
                raise ValueError(f"Target-language mismatch in {path}")
            if record.get("split", split) != split:
                raise ValueError(f"Split mismatch in {path}")
            record.update(
                run="__".join(parts),
                model=model,
                family=family,
                method=method,
                training=training,
                variant=variant,
                input_language=input_lang,
                target_language=target,
                split=split,
            )
            runs.append(record)
    return runs


def prompt_sensitivity(runs: list[dict[str, Any]], target: str) -> pd.DataFrame:
    # average over the matched english and target-language DEV sets
    frame = pd.DataFrame(
        [
            {
                "model": run["model"],
                "method": run["method"],
                "variant": run["variant"],
                "input_language": run["input_language"],
                "micro_f1": run["micro_f1"],
                "sample_jaccard": run["sample_jaccard"],
                "malformed_rate": run.get("malformed_rate"),
            }
            for run in runs
            if run["family"] == "prompting"
        ]
    )
    if frame.empty:
        return frame

    result = frame.groupby(["model", "method", "variant"], as_index=False).agg(
        mean_micro_f1=("micro_f1", "mean"),
        mean_sample_jaccard=("sample_jaccard", "mean"),
        mean_malformed_rate=("malformed_rate", "mean"),
        n_languages=("input_language", "nunique"),
    )
    result.insert(0, "target_language", target)
    return result.sort_values(["model", "method", "variant"])


def select_prompts(sensitivity: pd.DataFrame, target: str, report_dir: Path) -> dict[str, str]:
    # choose one wording per model and shot condition from DEV only
    if sensitivity.empty:
        raise ValueError("No DEV prompt sensitivity results found")

    selections = {}
    groups = list(sensitivity.groupby(["model", "method"]))
    if len(groups) != 4:
        raise ValueError("Prompt selection requires two models and two shot conditions")

    for (model, method), group in groups:
        if len(group) != 4 or not (group["n_languages"] == 2).all():
            raise ValueError(f"Incomplete DEV wording results for {model}/{method}")
        best = group.sort_values(
            ["mean_micro_f1", "mean_sample_jaccard", "variant"],
            ascending=[False, False, True],
        ).iloc[0]
        for input_lang in ("en", target):
            key = f"{model}__{target}__{input_lang}__{method}"
            selections[key] = str(best["variant"])

    # each target has its own selection file to avoid concurrent writes
    path = report_dir / f"selected_prompt_variants_{target}.json"
    path.write_text(json.dumps(selections, indent=2), encoding="utf-8")
    print(f"Saved DEV prompt choices to {path}")
    return selections


def choose_runs(runs: list[dict[str, Any]], choices: dict[str, str]) -> list[dict[str, Any]]:
    # retain all PEFT runs and the selected prompting variant for each condition
    selected = []
    for run in runs:
        if run["family"] == "peft":
            selected.append(run)
            continue
        key = f"{run['model']}__{run['target_language']}__{run['input_language']}__{run['method']}"
        if key not in choices:
            raise ValueError(f"Missing DEV prompt selection: {key}")
        if run["variant"] == choices[key]:
            selected.append(run)
    return selected


def validate_test_runs(runs: list[dict[str, Any]], target: str) -> None:
    # enforce the eight prompting and five PEFT evaluations required by the brief
    prompting = [r for r in runs if r["family"] == "prompting"]
    peft = [r for r in runs if r["family"] == "peft"]
    prompt_keys = {(r["model"], r["method"], r["input_language"]) for r in prompting}
    expected_peft = {
        ("english", "en"),
        ("english", target),
        ("target", target),
        ("bilingual", "en"),
        ("bilingual", target),
    }
    peft_keys = {(r["training"], r["input_language"]) for r in peft}
    if (
        len(prompting) != 8
        or len(prompt_keys) != 8
        or len({r["model"] for r in prompting}) != 2
        or len(peft) != 5
        or peft_keys != expected_peft
    ):
        raise ValueError(
            "Incomplete TEST results: expected 8 selected prompting runs "
            "(2 models x 2 methods x 2 languages) and 5 PEFT runs"
        )


def comparison_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{column: run.get(column) for column in RESULT_COLUMNS} for run in runs],
        columns=list(RESULT_COLUMNS),
    )


def per_emotion_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for run in runs:
        scores = run.get("per_label", {})
        for emotion in EMOTIONS:
            value = scores.get(emotion, {})
            rows.append(
                {
                    "run": run["run"],
                    "model": run["model"],
                    "family": run["family"],
                    "method": run["method"],
                    "training": run["training"],
                    "input_language": run["input_language"],
                    "emotion": emotion,
                    "precision": value.get("precision"),
                    "recall": value.get("recall"),
                    "f1": value.get("f1"),
                    "support": value.get("support"),
                }
            )
    return pd.DataFrame(rows)


def language_gaps(table: pd.DataFrame, target: str) -> pd.DataFrame:
    # english minus target micro f1 on matching experiment conditions
    frame = table.copy()
    frame["setting"] = frame.apply(
        lambda row: (
            f"{row['method']} / {row['variant']}"
            if row["family"] == "prompting"
            else f"peft / {row['training']}"
        ),
        axis=1,
    )
    pivot = frame.pivot_table(
        index=["model", "setting"],
        columns="input_language",
        values="micro_f1",
        aggfunc="first",
    )
    if "en" not in pivot or target not in pivot:
        return pd.DataFrame(columns=["model", "setting", "english_f1", "target_f1", "gap"])
    pivot = pivot.dropna(subset=["en", target]).reset_index()
    pivot = pivot.rename(columns={"en": "english_f1", target: "target_f1"})
    pivot["gap"] = pivot["english_f1"] - pivot["target_f1"]
    return pivot[["model", "setting", "english_f1", "target_f1", "gap"]]


def prediction_rows(project_root: Path, run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = (
        project_root / "outputs/predictions" / run["family"] / run["split"] / f"{run['run']}.jsonl"
    )
    if not path.exists():
        print(f"Warning: prediction file missing, skipping qualitative comparison: {path}")
        return {}
    records = read_rows(path)
    indexed = {str(row["alignment_id"]): row for row in records}
    if len(indexed) != len(records):
        raise ValueError(f"Duplicate alignment IDs in {path}")
    return indexed


def example_f1(gold: list[str], prediction: list[str]) -> float:
    # calculate a per-sentence f1 score to rank improvements and regressions
    correct = len(set(gold) & set(prediction))
    return 2 * correct / (len(gold) + len(prediction)) if gold or prediction else 1.0


def qualitative_examples(
    project_root: Path, runs: list[dict[str, Any]], max_examples: int = 10
) -> pd.DataFrame:
    # compare each PEFT adapter with zero-shot and five-shot Qwen on matched lines
    peft = [r for r in runs if r["family"] == "peft"]
    prompts = [r for r in runs if r["family"] == "prompting"]
    rows = []
    for trained in peft:
        trained_items = prediction_rows(project_root, trained)
        for baseline in prompts:
            if (
                trained["model"] != baseline["model"]
                or trained["input_language"] != baseline["input_language"]
            ):
                continue
            prompt_items = prediction_rows(project_root, baseline)
            for identifier in set(trained_items) & set(prompt_items):
                after, before = trained_items[identifier], prompt_items[identifier]
                if set(after["gold"]) != set(before["gold"]):
                    raise ValueError(f"Gold-label mismatch for aligned ID {identifier}")
                gold = after["gold"]
                change = example_f1(gold, after["pred"]) - example_f1(gold, before["pred"])
                if change == 0:
                    continue
                rows.append(
                    {
                        "alignment_id": identifier,
                        "input_language": trained["input_language"],
                        "training": trained["training"],
                        "prompt_method": baseline["method"],
                        "change": "improved" if change > 0 else "worsened",
                        "delta_f1": change,
                        "text": after["text"],
                        "gold": json.dumps(gold, ensure_ascii=False),
                        "prompt_prediction": json.dumps(before["pred"], ensure_ascii=False),
                        "peft_prediction": json.dumps(after["pred"], ensure_ascii=False),
                        "prompt_run": baseline["run"],
                        "peft_run": trained["run"],
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "alignment_id",
                "input_language",
                "training",
                "prompt_method",
                "change",
                "delta_f1",
                "text",
                "gold",
                "prompt_prediction",
                "peft_prediction",
                "prompt_run",
                "peft_run",
            ]
        )
    frame = pd.DataFrame(rows)
    # keep the strongest positive and negative changes per comparison
    groups = ["peft_run", "prompt_run", "change"]
    return (
        frame.assign(abs_delta=frame["delta_f1"].abs())
        .sort_values("abs_delta", ascending=False)
        .groupby(groups, group_keys=False)
        .head(max_examples)
        .drop(columns="abs_delta")
    )


def dataset_summary(project_root: Path, target: str) -> pd.DataFrame:
    # summarize projected reference labels and detect repeated split members
    folder = project_root / "data/processed" / target
    rows = []
    seen_ids: dict[str, str] = {}
    seen_text: dict[tuple[str, str], str] = {}
    for split in ("train", "dev", "test"):
        english_path = folder / f"{split}_en.jsonl"
        target_path = folder / f"{split}_{target}.jsonl"
        if not english_path.exists() or not target_path.exists():
            continue
        english = read_rows(english_path)
        assigned = read_rows(target_path)
        en_ids = {str(row["alignment_id"]) for row in english}
        target_ids = {str(row["alignment_id"]) for row in assigned}
        if en_ids != target_ids:
            raise ValueError(f"English and {target} alignments disagree on {split}")
        for language, examples in (("en", english), (target, assigned)):
            for row in examples:
                identifier = str(row["alignment_id"])
                text_key = (language, " ".join(row["input"].split()))
                if identifier in seen_ids and seen_ids[identifier] != split:
                    raise ValueError(f"Alignment ID appears in multiple splits: {identifier}")
                if text_key in seen_text and seen_text[text_key] != split:
                    raise ValueError(f"Exact subtitle appears in multiple splits: {text_key}")
                seen_ids[identifier] = split
                seen_text[text_key] = split
        labels = [gold_labels(row["output"]) for row in assigned]
        rows.append(
            {
                "target_language": target,
                "split": split,
                "n": len(labels),
                "mean_labels_per_example": sum(map(len, labels)) / len(labels),
                **{
                    emotion: sum(emotion in label_set for label_set in labels)
                    for emotion in EMOTIONS
                },
            }
        )
    return pd.DataFrame(rows)


def runtime_table(project_root: Path, runs: list[dict[str, Any]], target: str) -> pd.DataFrame:
    rows = []
    for run in runs:
        record = {
            "run": run["run"],
            "family": run["family"],
            "model": run["model"],
            "training": run["training"],
            "input_language": run["input_language"],
            "n": run.get("n"),
            "inference_seconds": run.get("inference_seconds"),
            "peak_gpu_memory_gb": run.get("peak_gpu_memory_gb"),
            "gpu": run.get("gpu"),
            "training_seconds": None,
        }
        if run["family"] == "peft":
            experiment = run["run"].split("__")[-1]
            metadata_path = (
                project_root
                / "outputs/models"
                / target
                / run["training"]
                / experiment
                / "metadata.json"
            )
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                record["training_seconds"] = metadata.get("training_seconds")
        rows.append(record)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["seconds_per_sample"] = pd.to_numeric(
            frame["inference_seconds"], errors="coerce"
        ) / pd.to_numeric(frame["n"], errors="coerce")
    return frame


def save_charts(
    table: pd.DataFrame,
    emotions: pd.DataFrame,
    sensitivity: pd.DataFrame,
    target: str,
    split: str,
    report_dir: Path,
) -> None:
    # generate comparison figures using stored metrics only
    for input_lang, part in table.groupby("input_language"):
        labels = [
            f"{row.model}\n{row.training if row.family == 'peft' else row.method}"
            for row in part.itertuples()
        ]
        fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.9), 4.5))
        ax.bar(range(len(labels)), part["micro_f1"])
        ax.set_xticks(range(len(labels)), labels=labels, rotation=60, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Micro F1")
        ax.set_title(f"{target.upper()} target — {str(input_lang).upper()} input — {split.upper()}")
        fig.tight_layout()
        fig.savefig(report_dir / f"{target}_{split}_micro_f1_{input_lang}.png", dpi=180)
        plt.close(fig)

        matrix = (
            emotions[emotions["input_language"] == input_lang]
            .pivot_table(index="run", columns="emotion", values="f1")
            .reindex(columns=EMOTIONS)
        )
        if not matrix.empty:
            fig, ax = plt.subplots(figsize=(10, max(4, len(matrix) * 0.45)))
            im = ax.imshow(matrix, vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(EMOTIONS)), labels=EMOTIONS, rotation=45, ha="right")
            ax.set_yticks(range(len(matrix)), labels=matrix.index)
            ax.set_title(f"Per-emotion F1 — {str(input_lang).upper()} — {split.upper()}")
            fig.colorbar(im, ax=ax, label="F1")
            fig.tight_layout()
            fig.savefig(report_dir / f"{target}_{split}_per_emotion_{input_lang}.png", dpi=180)
            plt.close(fig)

    if split == "dev" and not sensitivity.empty:
        labels = [f"{r.model}\n{r.method}\n{r.variant}" for r in sensitivity.itertuples()]
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.75), 4.5))
        ax.bar(range(len(labels)), sensitivity["mean_micro_f1"])
        ax.set_xticks(range(len(labels)), labels=labels, rotation=65, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean DEV Micro F1")
        fig.tight_layout()
        fig.savefig(report_dir / f"{target}_dev_prompt_sensitivity.png", dpi=180)
        plt.close(fig)


def generate_results(
    project_root: Path,
    target: str = "sv",
    split: str = "dev",
    select: bool = False,
    allow_incomplete: bool = False,
) -> None:
    """Build the report outputs from completed evaluation files"""
    if select and split != "dev":
        raise ValueError("Prompt selection is only allowed on DEV")
    if select and allow_incomplete:
        raise ValueError("Prompt selection requires the full DEV matrix")
    report_dir = project_root / "outputs/report"
    report_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(project_root, target, split)
    if not runs:
        raise FileNotFoundError(f"No {split.upper()} metric JSON files found for {target}")

    sensitivity = prompt_sensitivity(runs, target) if split == "dev" else pd.DataFrame()
    if split == "dev" and not sensitivity.empty:
        sensitivity.to_csv(report_dir / f"{target}_dev_prompt_sensitivity.csv", index=False)

    selections_path = report_dir / f"selected_prompt_variants_{target}.json"
    if select:
        selections = select_prompts(sensitivity, target, report_dir)
    elif selections_path.exists():
        selections = json.loads(selections_path.read_text(encoding="utf-8"))
    else:
        if split == "test":
            raise FileNotFoundError("Select prompt variants on DEV before analyzing TEST")
        selections = {}

    # keep selections for different assigned languages separate
    target_selections = {
        key: value
        for key, value in selections.items()
        if len(key.split("__")) == 4 and key.split("__")[1] == target
    }
    if split == "test" and not target_selections:
        raise ValueError(f"No DEV-selected prompt variants found for {target}")

    # show all DEV wordings until selection, then only the selected ones
    active_runs = choose_runs(runs, target_selections) if target_selections else runs
    if split == "test" and not allow_incomplete:
        validate_test_runs(active_runs, target)

    table = comparison_table(active_runs)
    per_label = per_emotion_table(active_runs)
    table.to_csv(report_dir / f"{target}_{split}_results.csv", index=False)
    per_label.to_csv(report_dir / f"{target}_{split}_per_emotion.csv", index=False)
    language_gaps(table, target).to_csv(
        report_dir / f"{target}_{split}_language_gaps.csv", index=False
    )
    qualitative_examples(project_root, active_runs).to_csv(
        report_dir / f"{target}_{split}_qualitative.csv", index=False
    )
    runtime_table(project_root, active_runs, target).to_csv(
        report_dir / f"{target}_{split}_runtime.csv", index=False
    )
    dataset = dataset_summary(project_root, target)
    if not dataset.empty:
        dataset.to_csv(report_dir / f"{target}_dataset_summary.csv", index=False)

    save_charts(table, per_label, sensitivity, target, split, report_dir)
    print(f"Saved {len(active_runs)} runs to {report_dir}")
    print(
        table[
            ["model", "method", "training", "input_language", "micro_f1", "sample_jaccard"]
        ].to_string(index=False)
    )


# report exports for the two target-language cohorts
# keep the german and swedish cohorts separate because their english subsets can differ

TARGETS = ("de", "sv")
PROMPT_VARIANTS = ("default", "paraphrase", "no_format_example", "terse")
SHOT_METHODS = ("zero_shot", "five_shot")
TARGET_NAMES = {"de": "German", "sv": "Swedish"}


def display_method(run: dict[str, Any]) -> str:
    if run["family"] == "prompting":
        return "Zero-shot" if run["method"] == "zero_shot" else "Five-shot"
    return {"english": "PEFT English", "target": "PEFT target", "bilingual": "PEFT bilingual"}.get(
        run["training"], f"PEFT {run['training']}"
    )


def display_model(model: str) -> str:
    return "EuroLLM" if "EuroLLM" in model else ("Qwen" if "Qwen" in model else model)


def target_records(runs: list[dict[str, Any]], target: str, language: str) -> list[dict[str, Any]]:
    return [r for r in runs if r["target_language"] == target and r["input_language"] == language]


def find_run(
    runs: list[dict[str, Any]],
    target: str,
    language: str,
    model_name: str,
    method: str,
    training: str = "none",
) -> dict[str, Any] | None:
    found = [
        r
        for r in target_records(runs, target, language)
        if display_model(r["model"]) == model_name
        and r["method"] == method
        and (method != "peft" or r["training"] == training)
    ]
    if len(found) > 1:
        raise ValueError(
            f"Duplicate results for {target}/{language}/{model_name}/{method}/{training}"
        )
    return found[0] if found else None


def latex_escape(value: object) -> str:
    # only escape literal text; numerical columns are formatted separately
    mapping = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(mapping.get(character, character) for character in str(value))


def number(value: object, digits: int = 3) -> str:
    # convert values safely, including numpy scalars and pandas missing values
    try:
        numeric = float(str(value))
    except (TypeError, ValueError):
        return "--"

    if not math.isfinite(numeric):
        return "--"

    return f"{numeric:.{digits}f}"


def write_latex_table(path: Path, columns: str, headers: list[str], rows: list[list[str]]) -> None:
    # export only a tabular so captions and labels stay in report/report.tex
    lines = [
        r"\begin{tabular}{" + columns + "}",
        r"\toprule",
        " & ".join(headers) + " " + "\\" * 2,
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " " + "\\" * 2)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def combined_main_table(runs: list[dict[str, Any]], folder: Path) -> None:
    # every english score belongs to its own aligned target cohort
    rows = []
    csv_rows = []
    methods = [
        ("Qwen", "zero_shot", "none"),
        ("Qwen", "five_shot", "none"),
        ("Qwen", "peft", "english"),
        ("Qwen", "peft", "target"),
        ("Qwen", "peft", "bilingual"),
        ("EuroLLM", "zero_shot", "none"),
        ("EuroLLM", "five_shot", "none"),
    ]
    for target in TARGETS:
        for model, method, training in methods:
            en = find_run(runs, target, "en", model, method, training)
            local = find_run(runs, target, target, model, method, training)
            if method == "peft" and training == "target":
                en = None  # not part of the required evaluation matrix
            label = display_method(
                local
                or en
                or {
                    "family": "peft" if method == "peft" else "prompting",
                    "method": method,
                    "training": training,
                }
            )
            rows.append(
                [
                    latex_escape(TARGET_NAMES[target]),
                    model,
                    latex_escape(label),
                    number(en.get("micro_f1") if en else None),
                    number(en.get("sample_jaccard") if en else None),
                    number(local.get("micro_f1") if local else None),
                    number(local.get("sample_jaccard") if local else None),
                ]
            )
            csv_rows.append(
                {
                    "cohort": target,
                    "model": model,
                    "method": label,
                    "en_micro_f1": en.get("micro_f1") if en else None,
                    "en_jaccard": en.get("sample_jaccard") if en else None,
                    "target_micro_f1": local.get("micro_f1") if local else None,
                    "target_jaccard": local.get("sample_jaccard") if local else None,
                    "en_macro_f1": en.get("macro_f1") if en else None,
                    "target_macro_f1": local.get("macro_f1") if local else None,
                }
            )
    pd.DataFrame(csv_rows).to_csv(folder / "combined_test_main_results.csv", index=False)
    write_latex_table(
        folder / "combined_test_main_table.tex",
        "lllrrrr",
        ["Cohort", "Model", "Method", "EN F1", "EN J", "Target F1", "Target J"],
        rows,
    )


def combined_dataset_table(project_root: Path, folder: Path) -> None:
    data = []
    for target in TARGETS:
        frame = dataset_summary(project_root, target)
        if frame.empty:
            continue
        data.extend(frame.to_dict("records"))
    frame = pd.DataFrame(data)
    frame.to_csv(folder / "combined_dataset_summary.csv", index=False)
    rows = []
    for target in TARGETS:
        for split in ("train", "dev", "test"):
            r = next(
                (
                    record
                    for record in data
                    if record["target_language"] == target and record["split"] == split
                ),
                None,
            )
            rows.append(
                [
                    latex_escape(TARGET_NAMES[target]),
                    split.upper(),
                    str(r["n"]) if r else "--",
                    number(r["mean_labels_per_example"], 2) if r else "--",
                ]
            )
    write_latex_table(
        folder / "combined_dataset_table.tex",
        "llrr",
        ["Cohort", "Split", "Aligned pairs", "Labels / item"],
        rows,
    )


def combined_transfer_table(runs: list[dict[str, Any]], folder: Path) -> None:
    rows, csv_rows = [], []
    for target in TARGETS:
        for model, method, training in [
            ("Qwen", "zero_shot", "none"),
            ("Qwen", "five_shot", "none"),
            ("Qwen", "peft", "english"),
            ("Qwen", "peft", "bilingual"),
        ]:
            en = find_run(runs, target, "en", model, method, training)
            local = find_run(runs, target, target, model, method, training)
            label = display_method(
                local
                or en
                or {
                    "family": "peft" if method == "peft" else "prompting",
                    "method": method,
                    "training": training,
                }
            )
            gap = float(en["micro_f1"]) - float(local["micro_f1"]) if en and local else None
            rows.append(
                [
                    latex_escape(TARGET_NAMES[target]),
                    latex_escape(label),
                    number(en.get("micro_f1") if en else None),
                    number(local.get("micro_f1") if local else None),
                    number(gap),
                ]
            )
            csv_rows.append(
                {
                    "cohort": target,
                    "method": label,
                    "en_micro_f1": en.get("micro_f1") if en else None,
                    "target_micro_f1": local.get("micro_f1") if local else None,
                    "en_minus_target": gap,
                }
            )
    pd.DataFrame(csv_rows).to_csv(folder / "combined_test_language_transfer.csv", index=False)
    write_latex_table(
        folder / "combined_test_language_transfer_table.tex",
        "llrrr",
        ["Cohort", "Qwen condition", "EN F1", "Target F1", "Gap"],
        rows,
    )


def combined_prompt_sensitivity(all_dev: list[dict[str, Any]], folder: Path) -> None:
    data = []
    for target in TARGETS:
        data.extend(
            prompt_sensitivity(
                [r for r in all_dev if r["target_language"] == target], target
            ).to_dict("records")
        )
    pd.DataFrame(data).to_csv(folder / "combined_dev_prompt_sensitivity.csv", index=False)
    if not data:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
    for i, target in enumerate(TARGETS):
        for j, model in enumerate(("Qwen", "EuroLLM")):
            ax = axes[i, j]
            arr = [[float("nan") for _ in PROMPT_VARIANTS] for _ in SHOT_METHODS]
            for r in data:
                if r["target_language"] != target or display_model(r["model"]) != model:
                    continue
                if r["method"] in SHOT_METHODS and r["variant"] in PROMPT_VARIANTS:
                    arr[SHOT_METHODS.index(r["method"])][PROMPT_VARIANTS.index(r["variant"])] = r[
                        "mean_micro_f1"
                    ]
            import numpy as np

            values = np.asarray(arr, dtype=float)
            ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
            ax.set_title(f"{TARGET_NAMES[target]} cohort · {model}")
            ax.set_yticks([0, 1], ["Zero-shot", "Five-shot"])
            ax.set_xticks(
                range(4), ["Default", "Paraphrase", "No example", "Terse"], rotation=22, ha="right"
            )
            for k in range(2):
                for l in range(4):
                    if np.isfinite(values[k, l]):
                        ax.text(
                            l,
                            k,
                            f"{values[k, l]:.3f}",
                            ha="center",
                            va="center",
                            color="white" if values[k, l] > 0.55 else "black",
                            fontsize=9,
                        )
    fig.savefig(folder / "combined_dev_prompt_sensitivity.png", dpi=190)
    plt.close(fig)


def combined_method_plot(runs: list[dict[str, Any]], folder: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, constrained_layout=True)
    configs = [
        ("Qwen", "zero_shot", "none"),
        ("Qwen", "five_shot", "none"),
        ("Qwen", "peft", "english"),
        ("Qwen", "peft", "target"),
        ("Qwen", "peft", "bilingual"),
        ("EuroLLM", "zero_shot", "none"),
        ("EuroLLM", "five_shot", "none"),
    ]
    labels = [
        "Qwen\nzero",
        "Qwen\nfive",
        "PEFT\nEN",
        "PEFT\ntarget",
        "PEFT\nbilingual",
        "EuroLLM\nzero",
        "EuroLLM\nfive",
    ]
    for ax, target in zip(axes, TARGETS, strict=True):
        for idx, (model, method, training) in enumerate(configs):
            run = find_run(runs, target, target, model, method, training)
            if run:
                ax.bar(idx, run["micro_f1"], color="steelblue" if model == "Qwen" else "darkgray")
            else:
                ax.text(idx, 0.04, "missing", rotation=90, ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_title(TARGET_NAMES[target] + " TEST")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Micro F1")
    fig.savefig(folder / "combined_test_method_comparison.png", dpi=190)
    plt.close(fig)


def combined_emotion_plot(runs: list[dict[str, Any]], folder: Path) -> None:
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    methods = [
        ("zero_shot", "none", "Zero-shot"),
        ("five_shot", "none", "Five-shot"),
        ("peft", "bilingual", "PEFT bilingual"),
    ]
    for ax, target in zip(axes, TARGETS, strict=True):
        matrix = np.full((len(EMOTIONS), len(methods)), np.nan)
        for j, (method, training, _) in enumerate(methods):
            run = find_run(runs, target, target, "Qwen", method, training)
            if not run:
                continue
            for i, emotion in enumerate(EMOTIONS):
                val = run.get("per_label", {}).get(emotion, {}).get("f1")
                if val is not None:
                    matrix[i, j] = float(val)
        ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(3), [m[2] for m in methods], rotation=18, ha="right", fontsize=9)
        ax.set_yticks(range(len(EMOTIONS)), EMOTIONS, fontsize=9)
        ax.set_title(TARGET_NAMES[target] + " TEST · Qwen")
        for i in range(len(EMOTIONS)):
            for j in range(3):
                if np.isfinite(matrix[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if matrix[i, j] > 0.55 else "black",
                    )
    fig.savefig(folder / "combined_test_per_emotion_f1.png", dpi=190)
    plt.close(fig)


def combined_qualitative_table(
    project_root: Path, runs: list[dict[str, Any]], folder: Path
) -> None:
    candidates = []
    for target in TARGETS:
        subset = [r for r in runs if r["target_language"] == target]
        frame = qualitative_examples(project_root, subset, max_examples=12)
        if frame.empty:
            continue
        frame.insert(0, "cohort", target)
        candidates.extend(frame.to_dict("records"))
    pd.DataFrame(candidates).to_csv(
        folder / "combined_test_qualitative_candidates.csv", index=False
    )
    rows = []
    for target in TARGETS:
        pool = [
            r
            for r in candidates
            if r["cohort"] == target
            and r["input_language"] == target
            and r["training"] == "bilingual"
            and r["prompt_method"] == "five_shot"
        ]
        for change in ("improved", "worsened"):
            options = sorted(
                (r for r in pool if r["change"] == change),
                key=lambda r: (-abs(r["delta_f1"]), str(r["alignment_id"])),
            )
            if not options:
                continue
            r = options[0]
            import textwrap

            text = textwrap.shorten(" ".join(str(r["text"]).split()), width=83, placeholder="...")
            rows.append(
                [
                    TARGET_NAMES[target],
                    latex_escape(change.capitalize()),
                    latex_escape(text),
                    latex_escape(r["gold"]),
                    latex_escape(r["prompt_prediction"]),
                    latex_escape(r["peft_prediction"]),
                ]
            )
    if not rows:
        (folder / "combined_test_qualitative_table.tex").unlink(missing_ok=True)
        return
    write_latex_table(
        folder / "combined_test_qualitative_table.tex",
        "llp{4cm}p{2.2cm}p{2.2cm}p{2.2cm}",
        ["Cohort", "Change", "Text", "Gold", "Five-shot", "PEFT"],
        rows,
    )


def combined_output_behavior_table(runs: list[dict[str, Any]], folder: Path) -> None:
    # compare the parser and the number of predicted labels on target-language test items
    comparisons = [
        ("Qwen", "zero_shot", "none"),
        ("Qwen", "five_shot", "none"),
        ("Qwen", "peft", "bilingual"),
        ("EuroLLM", "zero_shot", "none"),
        ("EuroLLM", "five_shot", "none"),
    ]
    rows = []
    csv_rows = []
    for target in TARGETS:
        for model, method, training in comparisons:
            run = find_run(runs, target, target, model, method, training)
            label = display_method(
                run
                or {
                    "family": "peft" if method == "peft" else "prompting",
                    "method": method,
                    "training": training,
                }
            )
            malformed = run.get("malformed_rate") if run else None
            empty = run.get("empty_prediction_rate") if run else None
            cardinality = run.get("pred_label_cardinality") if run else None
            rows.append(
                [
                    TARGET_NAMES[target],
                    model,
                    latex_escape(label),
                    number(float(malformed) * 100 if malformed is not None else None, 1),
                    number(float(empty) * 100 if empty is not None else None, 1),
                    number(cardinality, 2),
                ]
            )
            csv_rows.append(
                {
                    "cohort": target,
                    "model": model,
                    "method": label,
                    "malformed_rate": malformed,
                    "empty_prediction_rate": empty,
                    "pred_label_cardinality": cardinality,
                }
            )
    pd.DataFrame(csv_rows).to_csv(folder / "combined_test_output_behavior.csv", index=False)
    write_latex_table(
        folder / "combined_test_output_behavior_table.tex",
        "lllrrr",
        ["Cohort", "Model", "Method", r"Malformed \%", r"Empty \%", "Labels / item"],
        rows,
    )


def combined_runtime_table(project_root: Path, runs: list[dict[str, Any]], folder: Path) -> None:
    all_rows = []
    for target in TARGETS:
        local = [
            r for r in runs if r["target_language"] == target and r["input_language"] == target
        ]
        frame = runtime_table(project_root, local, target)
        if not frame.empty:
            frame.insert(0, "cohort", target)
            all_rows.extend(frame.to_dict("records"))
    pd.DataFrame(all_rows).to_csv(folder / "combined_test_runtime.csv", index=False)
    rows = []
    for target in TARGETS:
        for r in [x for x in all_rows if x["cohort"] == target]:
            label = (
                display_model(r["model"])
                + " "
                + (r["training"] if r["family"] == "peft" else "prompting")
            )
            rows.append(
                [
                    TARGET_NAMES[target],
                    latex_escape(label),
                    number(r.get("seconds_per_sample"), 3),
                    number(r.get("peak_gpu_memory_gb"), 1),
                ]
            )
    if rows:
        write_latex_table(
            folder / "combined_test_runtime_table.tex",
            "llrr",
            ["Cohort", "Run", "Seconds / item", "GPU GB"],
            rows,
        )


def generate_combined_report(
    project_root: Path,
    split: str,
    allow_incomplete: bool = False,
) -> None:
    """Export matching German/Swedish tables and figures without any new inference"""
    report_dir = project_root / "outputs/report"
    report_dir.mkdir(parents=True, exist_ok=True)
    all_runs = []
    for target in TARGETS:
        runs = load_runs(project_root, target, split)
        choices: dict[str, str] = {}
        if split == "test":
            selections_path = report_dir / f"selected_prompt_variants_{target}.json"
            if selections_path.exists():
                choices = json.loads(selections_path.read_text(encoding="utf-8"))
            if not choices:
                if allow_incomplete and not runs:
                    print(f"No {target.upper()} TEST runs yet; leaving its report cells blank")
                    continue
                raise ValueError(f"No DEV-selected prompt variants for {target}")
            runs = choose_runs(runs, choices)
            if not allow_incomplete:
                validate_test_runs(runs, target)
        all_runs.extend(runs)
    if not all_runs:
        raise FileNotFoundError(f"No saved {split.upper()} metrics found for de or sv")

    combined_dataset_table(project_root, report_dir)
    if split == "dev":
        combined_prompt_sensitivity(all_runs, report_dir)
        print(f"Saved German/Swedish DEV sensitivity to {report_dir}")
        return

    combined_main_table(all_runs, report_dir)
    combined_transfer_table(all_runs, report_dir)
    combined_method_plot(all_runs, report_dir)
    combined_emotion_plot(all_runs, report_dir)
    combined_qualitative_table(project_root, all_runs, report_dir)
    combined_output_behavior_table(all_runs, report_dir)
    combined_runtime_table(project_root, all_runs, report_dir)
    print(f"Saved combined German/Swedish TEST figures and tables to {report_dir}")
