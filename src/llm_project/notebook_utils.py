from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

LABELS = None  # optional: pass labels explicitly to functions


def binarise(lists: Sequence[Sequence[str]], labels: Sequence[str]) -> np.ndarray:
    label_set = list(labels)
    return np.array([[lab in set(x) for lab in label_set] for x in lists], dtype=bool)


def f1s(Y: np.ndarray, P: np.ndarray) -> tuple[float, float]:
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
) -> tuple[float, float, float, int]:
    ids = da.index.intersection(db.index)
    Ya = binarise(da.loc[ids, "gold"].tolist(), labels)
    Pa = binarise(da.loc[ids, "pred"].tolist(), labels)
    Yb = binarise(db.loc[ids, "gold"].tolist(), labels)
    Pb = binarise(db.loc[ids, "pred"].tolist(), labels)

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


def load_metrics(
    model: str, lang: str, condition: str, metrics_dir: str | Path, variant: str = "default"
) -> dict:
    path = Path(metrics_dir) / f"{model}__{lang}__{condition}__{variant}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_preds(
    model: str, lang: str, condition: str, pred_dir: str | Path, variant: str = "default"
) -> pd.DataFrame:
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
        rows[(r["model"], r["lang"], r["condition"])] = {
            l: met["per_label"][l]["f1"] for l in labels
        }
    df = pd.DataFrame(rows)
    first = summary.iloc[0]
    df["gold_support"] = [
        load_metrics(first["model"], first["lang"], first["condition"], metrics_dir)["per_label"][
            l
        ]["support"]
        for l in labels
    ]
    return df


def predicted_distribution(
    summary: pd.DataFrame, pred_dir: str | Path, labels: list[str]
) -> pd.DataFrame:
    rows = {}
    for _, r in summary.iterrows():
        df = load_preds(r["model"], r["lang"], r["condition"], pred_dir)
        rows[(r["model"], r["lang"], r["condition"])] = (
            pd.Series([l for p in df["pred"] for l in p])
            .value_counts()
            .reindex(labels, fill_value=0)
        )
    gold_counts = (
        pd.Series([l for g in df["gold"] for l in g]).value_counts().reindex(labels, fill_value=0)
    )
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


def show_errors(
    model: str, lang: str, condition: str, pred_dir: str | Path, k: int = 10, seed: int = 0
):
    """Display a sample of wrong predictions for qualitative analysis."""
    df = load_preds(model, lang, condition, pred_dir)
    wrong = df[df["gold"].apply(set) != df["pred"].apply(set)]
    print(f"{len(wrong)}/{len(df)} not exactly right")
    with pd.option_context("display.max_colwidth", 120):
        display(
            wrong.sample(min(k, len(wrong)), random_state=seed)[
                ["text", "gold", "pred", "raw_output"]
            ]
        )


def get_labels(df: pd.DataFrame, index, column: str) -> list[str]:
    # read a list of emotion labels from the dataframe
    value: object = df.at[index, column]

    if not isinstance(value, list):
        raise TypeError(f"Expected a list in column '{column}'")

    labels = []
    for label in value:
        if not isinstance(label, str):
            raise TypeError("Emotion labels must be strings")
        labels.append(label)

    return labels


def ensemble_majority(
    pred_dfs: dict[str, pd.DataFrame], labels: Iterable[str], tie_strategy: str = "union"
) -> pd.DataFrame:
    """Create an ensemble prediction by majority vote across provided prediction DataFrames.

    pred_dfs: mapping name -> DataFrame containing at least columns `gold` and `pred` and indexed by ID.
    labels: ordered list of all labels.
    tie_strategy: 'union'|'first'|'max' behaviour when no label reaches strict majority.
    Returns a DataFrame with the same index and columns `gold` and `pred` (list of labels).
    """
    dfs = list(pred_dfs.values())
    if not dfs:
        return pd.DataFrame(columns=["gold", "pred"])  # empty
    # align on the intersection of IDs
    ids = dfs[0].index
    for df in dfs[1:]:
        ids = ids.intersection(df.index)

    out_rows = []
    nsys = len(dfs)
    for i in ids:
        gold = get_labels(dfs[0], i, "gold")
        votes: dict[str, int] = {}

        for df in dfs:
            pred = get_labels(df, i, "pred")

            for lab in pred:
                votes[lab] = votes.get(lab, 0) + 1
        # labels reaching strict majority (> nsys/2)
        selected = [l for l, c in votes.items() if c > nsys / 2]
        if not selected:
            if tie_strategy == "union":
                selected = sorted(votes.keys())
            elif tie_strategy == "first":
                selected = get_labels(dfs[0], i, "pred")
            else:  # max: choose labels with max count
                if votes:
                    maxc = max(votes.values())
                    selected = sorted([l for l, c in votes.items() if c == maxc])
                else:
                    selected = []
        out_rows.append({"id": i, "gold": gold, "pred": selected})

    out = pd.DataFrame(out_rows).set_index("id")
    return out


def score_predictions(df: pd.DataFrame, labels: Sequence[str]) -> dict:
    """Compute simple summary metrics for a predictions DataFrame with `gold` and `pred` columns.

    Returns a dict with `macro_f1`, `micro_f1`, `exact_match`, `jaccard`,
    `pred_label_cardinality`, `gold_label_cardinality`.
    """
    if df.empty:
        return {}
    Y = binarise(df["gold"].tolist(), labels)
    P = binarise(df["pred"].tolist(), labels)
    macro, micro = f1s(Y, P)
    # exact match
    exact = np.mean([set(g) == set(p) for g, p in zip(df["gold"], df["pred"])])
    # jaccard per instance
    jacc = []
    for g, p in zip(df["gold"], df["pred"]):
        gs, ps = set(g), set(p)
        jacc.append(len(gs & ps) / max(1, len(gs | ps)))
    return {
        "macro_f1": float(macro),
        "micro_f1": float(micro),
        "exact_match": float(exact),
        "jaccard": float(np.mean(jacc)),
        "pred_label_cardinality": float(np.mean([len(p) for p in df["pred"]])),
        "gold_label_cardinality": float(np.mean([len(g) for g in df["gold"]])),
    }


def text_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    """Compute the soft F1-like score used in the qualitative comparison cells."""
    g, p = set(gold), set(pred)
    if not g and not p:
        return 1.0
    return 2 * len(g & p) / max(len(g) + len(p), 1)


def compare_predictions(
    base_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    *,
    base_name: str = "base",
    candidate_name: str = "candidate",
    k: int = 10,
    english_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare two prediction DataFrames and summarize where the candidate helps/hurts.

    The two frames must have aligned indexes and contain at least `text`, `gold`, and `pred`.
    Positive `delta` means the candidate is better than the base model on that example.
    """
    ids = base_df.index.intersection(candidate_df.index)
    if ids.empty:
        return pd.DataFrame(columns=["text", "gold", base_name, candidate_name, "delta"])

    cmp = pd.DataFrame(
        {
            "text": base_df.loc[ids, "text"],
            "gold": base_df.loc[ids, "gold"],
            base_name: base_df.loc[ids, "pred"],
            candidate_name: candidate_df.loc[ids, "pred"],
        }
    )
    # calculate the change in f1 for each example
    cmp["delta"] = [
        text_f1(g, q) - text_f1(g, p)
        for g, p, q in zip(
            cmp["gold"],
            cmp[base_name],
            cmp[candidate_name],
            strict=True,
        )
    ]

    # include english translations if available
    if english_df is not None:
        cmp["english_text"] = english_df.reindex(ids)["text"]

    # summarize improvements and regressions
    improved = (cmp["delta"] > 0).sum()
    worsened = (cmp["delta"] < 0).sum()
    same = (cmp["delta"] == 0).sum()

    print(f"{candidate_name} vs {base_name}")
    print(f"Improved: {improved}, worsened: {worsened}, unchanged: {same}")

    # show examples with the largest differences
    with pd.option_context("display.max_colwidth", 80):
        print("Candidate IMPROVED:")
        display(cmp.sort_values("delta", ascending=False).head(k))

        print("Candidate WORSENED:")
        display(cmp.sort_values("delta", ascending=True).head(k))

    return cmp
