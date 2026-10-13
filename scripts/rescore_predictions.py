from __future__ import annotations
 
import argparse
import glob
import json
import sys
from pathlib import Path
 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
 
from llm_project.prompt_evaluation import compute_metrics, confusion_matrices  # noqa: E402
from llm_project.prompting import gold_labels, parse_prediction  # noqa: E402
 
 
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help='glob, e.g. "outputs/peft_raw/test/*.jsonl"')
    ap.add_argument("--family", default="peft", help="output subfolder, e.g. peft")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--id-field", default="alignment_id")
    ap.add_argument("--gold-field", default="gold")
    ap.add_argument("--raw-field", default="raw_output")
    ap.add_argument("--text-field", default="text")
    args = ap.parse_args()
 
    files = sorted(glob.glob(args.inputs))
    if not files:
        sys.exit(f"No files match {args.inputs}")
 
    pred_dir = PROJECT_ROOT / "outputs" / "predictions" / args.family / args.split
    metrics_dir = PROJECT_ROOT / "outputs" / "metrics" / args.family / args.split
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
 
    for path in files:
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        missing = {args.id_field, args.gold_field, args.raw_field} - set(rows[0])
        if missing:
            sys.exit(f"{path}: missing fields {missing}; available: {sorted(rows[0])}")
 
        ids = [r[args.id_field] for r in rows]
        if len(set(ids)) != len(ids):
            sys.exit(f"{path}: duplicate IDs; each test text must appear exactly once")
 
        gold = [gold_labels(r[args.gold_field]) for r in rows]
        parsed = [parse_prediction(str(r[args.raw_field])) for r in rows]
        pred = [list(p.labels) for p in parsed]
        metrics = compute_metrics(gold, pred, parsed)
        metrics["confusion"] = confusion_matrices(gold, pred)
 
        run = Path(path).stem
        with open(pred_dir / f"{run}.jsonl", "w", encoding="utf-8") as f:
            for r, g, p, pr in zip(rows, gold, pred, parsed):
                f.write(json.dumps({
                    args.id_field: r[args.id_field],
                    "text": r.get(args.text_field, ""),
                    "gold": g, "pred": p,
                    "raw_output": r[args.raw_field], "malformed": pr.malformed,
                }, ensure_ascii=False) + "\n")
        (metrics_dir / f"{run}.json").write_text(json.dumps({
            "run": run, "source_file": str(path), "scored_with": "parse_prediction + compute_metrics",
            **metrics,
        }, indent=2), encoding="utf-8")
        print(f"{run}: n={metrics['n']} macro_f1={metrics['macro_f1']:.3f} "
              f"micro_f1={metrics['micro_f1']:.3f} malformed={metrics['malformed_rate']:.3f}")
 
 
if __name__ == "__main__":
    main()