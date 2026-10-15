import argparse

from llm_project.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lang", choices=["de", "sv"])
    parser.add_argument("condition", choices=["en", "target", "bilingual"])
    args = parser.parse_args()
    train(args.lang, args.condition)
