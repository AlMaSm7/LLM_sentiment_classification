import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(records, path):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_mixed(language: str, split: str, seed: int):
    data_dir = PROJECT_ROOT / "data" / "processed" / language
    en_rows = load_jsonl(data_dir / f"{split}_en.jsonl")
    tgt_rows = load_jsonl(data_dir / f"{split}_{language}.jsonl")

    en_by_id = {r["alignment_id"]: r for r in en_rows}
    tgt_by_id = {r["alignment_id"]: r for r in tgt_rows}

    common_ids = sorted(set(en_by_id) & set(tgt_by_id))
    missing = (set(en_by_id) | set(tgt_by_id)) - set(common_ids)
    if missing:
        print(
            f"[{split}] Warning: {len(missing)} alignment_ids not present in both "
            f"languages, skipping those."
        )

    rng = random.Random(seed)
    mixed = []
    for aid in common_ids:
        chosen = en_by_id[aid] if rng.random() < 0.5 else tgt_by_id[aid]
        mixed.append(chosen)

    out_path = data_dir / f"{split}_mixed.jsonl"
    write_jsonl(mixed, out_path)
    print(f"[{split}] wrote {len(mixed)} examples -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", nargs="+", default=["train"])
    args = parser.parse_args()

    for split in args.splits:
        make_mixed(args.language, split, args.seed)


if __name__ == "__main__":
    main()
