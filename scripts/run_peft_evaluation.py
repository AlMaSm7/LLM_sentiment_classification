import argparse

from llm_project.peft_evaluation import run_peft_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PEFT adapter")
    parser.add_argument("--language", required=True, help="assigned language, e.g. sv")
    parser.add_argument(
        "--condition",
        required=True,
        choices=["english", "target", "bilingual"],
        help="language condition used to train the adapter",
    )
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--experiment", help="experiment name (default: configs/peft.yaml)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_peft_evaluation(
        language=args.language,
        condition=args.condition,
        split=args.split,
        experiment=args.experiment,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
