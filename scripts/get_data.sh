#!/usr/bin/env bash
set -euo pipefail

# requirement 3 - download xed labels and original opus raw alignments
# use LANGUAGES="de" bash get_data.sh for just german
LANGUAGES="${LANGUAGES:-de sv}"
XED="https://raw.githubusercontent.com/Helsinki-NLP/XED/master/Projections"
CACHE="data/raw/opus_cache"

mkdir -p data/raw "$CACHE"

for lang in $LANGUAGES; do
    if [[ "$lang" != de && "$lang" != sv ]]; then
        echo "unsupported language: $lang" >&2
        exit 1
    fi

    echo "getting xed labels for $lang"
    # write to a temporary file so interrupted downloads never corrupt the source
    curl -fL --retry 3 "$XED/${lang}-projections.tsv" -o "data/raw/${lang}-projections.tsv.tmp"
    mv "data/raw/${lang}-projections.tsv.tmp" "data/raw/${lang}-projections.tsv"

    echo "listing available opus OpenSubtitles v2018 raw resources"
    uv run opus_get -d OpenSubtitles -r v2018 -s en -t "$lang" -p raw -l

    echo "downloading opus raw text and alignment files into one cache"
    uv run opus_get -d OpenSubtitles -r v2018 -s en -t "$lang" -p raw -dl "$CACHE" -q

    echo "exporting parallel sentences and original opus sentence ids"
    # requirement 3 - one-to-one alignments and sentence provenance
    # -id is a separate output file, never the xed labels file
    uv run opus_read -d OpenSubtitles -r v2018 -s en -t "$lang" \
        -p raw -S 1 -T 1 -wm moses \
        -w "data/raw/en-${lang}.en" "data/raw/en-${lang}.${lang}" \
        -id "data/raw/en-${lang}.ids" -dl "$CACHE" -q

done

echo "data ready - inspect en-*.ids then run uv run python prepare.py --lang de --candidates"
