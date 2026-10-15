#!/bin/bash
set -euo pipefail

# clean xed projection files for german and swedish
for lang in de sv; do
    file="data/raw/${lang}-projections.tsv"

    # skip missing files
    if [[ ! -f "$file" ]]; then
        echo "Skipping $lang: file not found"
        continue
    fi

    # back up the original file
    cp "$file" "${file%.tsv}.bak.tsv"

    # keep only rows with exactly two tab-separated columns
    awk -F '\t' 'NF == 2' "$file" > "${file%.tsv}.clean.tsv"

    # replace the original file
    mv "${file%.tsv}.clean.tsv" "$file"

    # show remaining rows
    echo "$lang: $(wc -l < "$file") valid-format rows remaining"
done
