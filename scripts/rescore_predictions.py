from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm_project.prompt_evaluation import compute_metrics, confusion_matrices
from llm_project.prompting import gold_labels, parse_prediction


# path: the prediction file to score
# args: the cmd line opts, like alignment_id, gold, raw_output, text
# pred_dir: the output dir for the scored predictions
# metrics_dir: the output dir for the metrics json file
def rescore_file(path, args, pred_dir, metrics_dir):
    # read the file
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]

    missing = {args.id_field, args.gold_field, args.raw_field} - set(rows[0])
    if missing:
        sys.exit(f"{path} is missing the fields {missing}. It has: {sorted(rows[0])}")

    ids = [row[args.id_field] for row in rows]
    if len(ids) != len(set(ids)):
        sys.exit(f"{path} contains the same text more than once.")

    gold = [gold_labels(row[args.gold_field]) for row in rows]
    parsed = [parse_prediction(str(row[args.raw_field])) for row in rows]
    pred = [list(p.labels) for p in parsed]

    metrics = compute_metrics(gold, pred, parsed)
    metrics["confusion"] = confusion_matrices(gold, pred)

    run = Path(path).stem
    with open(pred_dir / f"{run}.jsonl", "w", encoding="utf-8") as f:
        for row, g, p, parse in zip(rows, gold, pred, parsed):
            record = {
                args.id_field: row[args.id_field],
                "text": row.get(args.text_field, ""),
                "gold": g,
                "pred": p,
                "raw_output": row[args.raw_field],
                "malformed": parse.malformed,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics_file = metrics_dir / f"{run}.json"
    metrics_file.write_text(
        json.dumps(
            {
                "run": run,
                "source_file": str(path),
                "scored_with": "parse_prediction + compute_metrics",
                **metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"{run}: n={metrics['n']} macro_f1={metrics['macro_f1']:.3f} "
        f"micro_f1={metrics['micro_f1']:.3f} malformed={metrics['malformed_rate']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs", required=True, help='glob, e.g. "outputs/peft_raw/test/*.jsonl"'
    )
    parser.add_argument("--family", default="peft", help="output subfolder, e.g. peft")
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--id-field", default="alignment_id")
    parser.add_argument("--gold-field", default="gold")
    parser.add_argument("--raw-field", default="raw_output")
    parser.add_argument("--text-field", default="text")
    args = parser.parse_args()

    files = sorted(glob.glob(args.inputs))
    if not files:
        sys.exit(f"No files match {args.inputs}")

    pred_dir = PROJECT_ROOT / "outputs" / "predictions" / args.family / args.split
    metrics_dir = PROJECT_ROOT / "outputs" / "metrics" / args.family / args.split
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        rescore_file(path, args, pred_dir, metrics_dir)


if __name__ == "__main__":
    main()
