import argparse

from llm_project.prepare import LANGUAGES, make_candidates, split_language

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=LANGUAGES, required=True)
    parser.add_argument("--candidates", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.candidates:
        make_candidates(args.lang)
    else:
        split_language(args.lang, args.force)
