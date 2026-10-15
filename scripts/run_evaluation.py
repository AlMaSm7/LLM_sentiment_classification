import argparse

from llm_project.evaluate import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["prompt", "peft"])
    parser.add_argument("target_language", choices=["de", "sv"])
    parser.add_argument("input_language", choices=["en", "target"])
    parser.add_argument("--model", choices=["qwen", "phi"], default="qwen")
    parser.add_argument(
        "--condition", choices=["none", "en", "target", "bilingual"], default="none"
    )
    parser.add_argument("--shots", type=int, choices=[0, 5], default=0)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--version", choices=["auto", "1", "2"], default="1")
    parser.add_argument("--overwrite", action="store_true")
    evaluate(parser.parse_args())
