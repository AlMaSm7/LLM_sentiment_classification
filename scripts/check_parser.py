"""Check that the current prompting.py parses exactly like an older, tested version.

    git show ca162be:src/llm_project/prompting.py > /tmp/old_prompting.py
    python scripts/check_parser.py /tmp/old_prompting.py

Compares the prompt fingerprint, parse_prediction and gold_labels on every
PEFT raw output in outputs/predictions/peft_raw/dev/.
"""

import glob
import importlib.util
import json
import sys

sys.path.insert(0, "src")
from llm_project import prompting as new  # noqa: E402

old_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/old_prompting.py"
spec = importlib.util.spec_from_file_location("old_prompting", old_path)
old = importlib.util.module_from_spec(spec)
sys.modules["old_prompting"] = old  # dataclasses need the module to be registered
spec.loader.exec_module(old)

print("fingerprint old:", old.prompt_fingerprint())
print("fingerprint new:", new.prompt_fingerprint())


def as_tuple(parsed):
    return (tuple(parsed.labels), parsed.malformed, parsed.strict_json)


total = parse_diff = gold_diff = 0
for path in sorted(glob.glob("outputs/predictions/peft_raw/dev/*.jsonl")):
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        total += 1
        if as_tuple(new.parse_prediction(row["raw_output"])) != as_tuple(
            old.parse_prediction(row["raw_output"])
        ):
            parse_diff += 1
            if parse_diff <= 3:
                print("parse differs:", repr(row["raw_output"][:80]))
        if list(new.gold_labels(row["gold"])) != list(old.gold_labels(row["gold"])):
            gold_diff += 1

print(f"{total} rows, parse differences: {parse_diff}, gold differences: {gold_diff}")
