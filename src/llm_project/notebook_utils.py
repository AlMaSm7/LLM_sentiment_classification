from __future__ import annotations
from typing import Sequence, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display

LABELS = None  # optional: pass labels explicitly to functions


def binarise(lists: Sequence[Sequence[str]], labels: Sequence[str]) -> np.ndarray:
    label_set = list(labels)
    return np.array([[lab in set(x) for lab in label_set] for x in lists], dtype=bool)


def f1s(Y: np.ndarray, P: np.ndarray) -> Tuple[float, float]:
    tp = (Y & P).sum(0)
    fp = (~Y & P).sum(0)
    fn = (Y & ~P).sum(0)
    d = 2 * tp + fp + fn
    macro = np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0).mean()
    micro = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)
    return float(macro), float(micro)


def paired_bootstrap(
    da: pd.DataFrame,
    db: pd.DataFrame,
    labels: Sequence[str],
    n_boot: int = 1000,
    seed: int = 0,
) -> Tuple[float, float, float, int]:
    ids = da.index.intersection(db.index)
    Ya = binarise(da.loc[ids, "gold"], labels)
    Pa = binarise(da.loc[ids, "pred"], labels)
    Yb = binarise(db.loc[ids, "gold"], labels)
    Pb = binarise(db.loc[ids, "pred"], labels)

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(ids), len(ids))
        diffs.append(f1s(Yb[i], Pb[i])[0] - f1s(Ya[i], Pa[i])[0])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return f1s(Yb, Pb)[0] - f1s(Ya, Pa)[0], float(lo), float(hi), len(ids)


def load_summary(metrics_dir: str | Path) -> pd.DataFrame:
    """Load the `summary.csv` produced alongside prompting runs."""
    path = Path(metrics_dir) / "summary.csv"
    return pd.read_csv(path)


def load_metrics(model: str, lang: str, condition: str, metrics_dir: str | Path, variant: str = "default") -> dict:
    path = Path(metrics_dir) / f"{model}__{lang}__{condition}__{variant}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_preds(model: str, lang: str, condition: str, pred_dir: str | Path, variant: str = "default") -> pd.DataFrame:
    path = Path(pred_dir) / f"{model}__{lang}__{condition}__{variant}.jsonl"
    return pd.DataFrame([json.loads(l) for l in path.open(encoding="utf-8")])


def main_table(summary: pd.DataFrame, cols=None) -> pd.DataFrame:
    cols = cols or ["macro_f1", "micro_f1", "samples_f1", "jaccard", "exact_match"]
    tbl = summary.pivot_table(index=["model", "lang"], columns="condition", values=cols)
    return tbl.reindex(columns=["zero_shot", "five_shot"], level=1)


def gain_and_gap(summary: pd.DataFrame) -> pd.DataFrame:
    m = summary.set_index(["model", "lang", "condition"])["macro_f1"].unstack("condition")
    gain = (m["five_shot"] - m["zero_shot"]).unstack("lang").add_prefix("five_shot_gain_")
    m2 = summary.set_index(["model", "condition", "lang"])["macro_f1"].unstack("lang")
    langs = [c for c in m2.columns if c != "en"]
    gap = m2["en"].sub(m2[langs[0]]).unstack("condition").add_prefix("en_minus_" + langs[0] + "_")
    return pd.concat([gain, gap], axis=1)


def format_df(summary: pd.DataFrame) -> pd.DataFrame:
    fmt_cols = [
        "malformed_rate",
        "strict_json_rate",
        "wellformed_but_no_label_rate",
        "empty_prediction_rate",
    ]
    return summary.set_index(["model", "lang", "condition"])[fmt_cols]


def per_label_f1(summary: pd.DataFrame, metrics_dir: str | Path, labels: list[str]) -> pd.DataFrame:
    rows = {}
    for _, r in summary.iterrows():
        met = load_metrics(r["model"], r["lang"], r["condition"], metrics_dir)
        rows[(r["model"], r["lang"], r["condition"])]= {l: met["per_label"][l]["f1"] for l in labels}
    df = pd.DataFrame(rows)
    first = summary.iloc[0]
    df["gold_support"] = [
        load_metrics(first["model"], first["lang"], first["condition"], metrics_dir)["per_label"][l]["support"]
        for l in labels
    ]
    return df


def predicted_distribution(summary: pd.DataFrame, pred_dir: str | Path, labels: list[str]) -> pd.DataFrame:
    rows = {}
    for _, r in summary.iterrows():
        df = load_preds(r["model"], r["lang"], r["condition"], pred_dir)
        rows[(r["model"], r["lang"], r["condition"])]= pd.Series([l for p in df["pred"] for l in p]).value_counts().reindex(labels, fill_value=0)
    gold_counts = pd.Series([l for g in df["gold"] for l in g]).value_counts().reindex(labels, fill_value=0)
    dist = pd.DataFrame(rows)
    dist.insert(0, "GOLD", gold_counts)
    return dist


def empty_predictions_summary(summary: pd.DataFrame, pred_dir: str | Path) -> list[tuple]:
    out = []
    for _, r in summary.iterrows():
        df = load_preds(r["model"], r["lang"], r["condition"], pred_dir)
        empty = df[df["pred"].str.len() == 0]
        counts = pd.Series([l for g in empty["gold"] for l in g]).value_counts().to_dict()
        out.append((r["model"], r["lang"], r["condition"], len(empty), len(df), counts))
    return out


def show_errors(model: str, lang: str, condition: str, pred_dir: str | Path, k: int = 10, seed: int = 0):
    """Display a sample of wrong predictions for qualitative analysis."""
    df = load_preds(model, lang, condition, pred_dir)
    wrong = df[df["gold"].apply(set) != df["pred"].apply(set)]
    print(f"{len(wrong)}/{len(df)} not exactly right")
    with pd.option_context("display.max_colwidth", 120):
        display(wrong.sample(min(k, len(wrong)), random_state=seed)[["text", "gold", "pred", "raw_output"]])
