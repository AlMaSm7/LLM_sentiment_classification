"""Select prompts on DEV, then make reproducible tables and error examples."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

METRICS = Path("outputs/metrics")
PREDICTIONS = Path("outputs/predictions")
REPORT = Path("outputs/report")
SELECTION = Path("outputs/prompt_selection.json")
LANGUAGES = ("de", "sv")
INPUT_LANGUAGES = ("en", "target")
MODELS = ("qwen", "phi")
SHOTS = (0, 5)


def load_metrics(split):
    rows = []
    for path in sorted(METRICS.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["split"] == split:
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No {split} result files found in {METRICS}")
    return rows


def select_prompts(rows, langs):
    selected = {}
    sensitivity = []
    for lang in langs:
        for model in MODELS:
            for shots in SHOTS:
                options = []
                for version in (1, 2):
                    matches = [
                        r
                        for r in rows
                        if r["lang"] == lang
                        and r["method"] == "prompt"
                        and r["model"] == model
                        and r["shots"] == shots
                        and r["prompt_version"] == version
                    ]
                    if {r["input_language"] for r in matches} != set(INPUT_LANGUAGES) or len(
                        matches
                    ) != 2:
                        raise ValueError(
                            f"Missing DEV scores: {lang} {model}, {shots} shots, v{version}"
                        )
                    mean_f1 = sum(r["micro_f1"] for r in matches) / 2
                    mean_j = sum(r["sample_jaccard"] for r in matches) / 2
                    options.append((mean_f1, mean_j, -version, version))
                    sensitivity.append(
                        {
                            "lang": lang,
                            "model": model,
                            "shots": shots,
                            "version": version,
                            "mean_micro_f1": mean_f1,
                            "mean_sample_jaccard": mean_j,
                        }
                    )
                selected[f"{lang}_{model}_s{shots}"] = max(options)[-1]
    REPORT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sensitivity).to_csv(REPORT / "dev_prompt_sensitivity.csv", index=False)
    SELECTION.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(SELECTION.read_text(encoding="utf-8")) if SELECTION.exists() else {}
    previous.update(selected)
    SELECTION.write_text(json.dumps(previous, indent=2), encoding="utf-8")
    print(f"Saved DEV-selected versions to {SELECTION}")
    print("Do not change the selection after looking at TEST scores")


def required_results(rows, langs):
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    filtered = []
    for lang in langs:
        expected = []
        for model in MODELS:
            for shots in SHOTS:
                version = selection[f"{lang}_{model}_s{shots}"]
                for language in INPUT_LANGUAGES:
                    expected.append(("prompt", model, "none", language, shots, version))
        for condition, languages in (
            ("en", INPUT_LANGUAGES),
            ("target", ("target",)),
            ("bilingual", INPUT_LANGUAGES),
        ):
            for language in languages:
                expected.append(("peft", "qwen", condition, language, 0, 1))
        for method, model, condition, language, shots, version in expected:
            matches = [
                r
                for r in rows
                if (
                    r["lang"],
                    r["method"],
                    r["model"],
                    r["condition"],
                    r["input_language"],
                    r["shots"],
                    r["prompt_version"],
                )
                == (lang, method, model, condition, language, shots, version)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one TEST run for {lang}: "
                    f"{method} {model} {condition} {language} shots={shots} version={version}"
                )
            filtered.extend(matches)
    return filtered


def read_predictions(run):
    path = PREDICTIONS / f"{run}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        x["id"]: x
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for x in [json.loads(line)]
    }


def create_report(rows, split):
    REPORT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        [
            {
                k: v
                for k, v in r.items()
                if k
                not in {
                    "per_emotion",
                    "prompt_system",
                    "prompt_user",
                    "decoding",
                    "demo_alignment_ids",
                }
            }
            for r in rows
        ]
    )
    table = table.sort_values(["lang", "input_language", "method", "model", "condition", "shots"])
    table.to_csv(REPORT / f"{split}_results.csv", index=False)
    per_emotion = [
        {
            "run": r["run"],
            "lang": r["lang"],
            "input_language": r["input_language"],
            "emotion": name,
            **scores,
        }
        for r in rows
        for name, scores in r["per_emotion"].items()
    ]
    pd.DataFrame(per_emotion).to_csv(REPORT / f"{split}_per_emotion.csv", index=False)

    for (lang, language), part in table.groupby(["lang", "input_language"]):
        names = [f"{r.model} {r.method} {r.condition} {r.shots}-shot" for r in part.itertuples()]
        fig, ax = plt.subplots(figsize=(max(9, len(names) * 1.15), 5))
        ax.bar(range(len(names)), part["micro_f1"])
        ax.set_xticks(range(len(names)), names, rotation=60, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Micro F1")
        ax.set_title(
            f"{lang}: {language} test data" if split == "test" else f"{lang}: {language} DEV"
        )
        fig.tight_layout()
        fig.savefig(REPORT / f"{split}_micro_f1_{lang}_{language}.png", dpi=160)
        plt.close(fig)

    if split == "test":
        changes = []
        for lang in sorted(table["lang"].unique()):
            for language in INPUT_LANGUAGES:
                subset = [r for r in rows if r["lang"] == lang and r["input_language"] == language]
                baseline = next(
                    (
                        r
                        for r in subset
                        if r["method"] == "prompt" and r["model"] == "qwen" and r["shots"] == 0
                    ),
                    None,
                )
                if baseline is None:
                    continue
                originals = read_predictions(baseline["run"])
                for result in (r for r in subset if r["method"] == "peft"):
                    trained = read_predictions(result["run"])
                    if originals.keys() != trained.keys():
                        raise ValueError("Cannot compare different test IDs across methods")
                    for example_id, previous in originals.items():
                        current = trained[example_id]
                        if previous["gold"] != current["gold"]:
                            raise ValueError(f"Mismatching gold labels for {example_id}")
                        before = set(previous["prediction"]) == set(previous["gold"])
                        after = set(current["prediction"]) == set(current["gold"])
                        if before == after:
                            continue
                        changes.append(
                            {
                                "lang": lang,
                                "language": language,
                                "condition": result["condition"],
                                "change": "improved" if after else "worsened",
                                "id": example_id,
                                "text": current["text"],
                                "gold": json.dumps(current["gold"]),
                                "zero_shot": json.dumps(previous["prediction"]),
                                "peft": json.dumps(current["prediction"]),
                            }
                        )
        pd.DataFrame(
            changes,
            columns=[
                "lang",
                "language",
                "condition",
                "change",
                "id",
                "text",
                "gold",
                "zero_shot",
                "peft",
            ],
        ).to_csv(REPORT / "qualitative.csv", index=False)

    print(f"Saved {split} tables and figures in {REPORT}")
