import argparse
from pathlib import Path

from lithonlp.predict import DrillCoreClassifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict input text using the drill core classification model."
    )
    parser.add_argument("text")
    parser.add_argument(
        "-p",
        "--bundled-model-path",
        required=True,
        type=Path,
        help="Path to the input bundled model.",
    )
    parser.add_argument(
        "-m",
        "--per-target-model-names",
        nargs="+",
        default=["HL_cor", "NL1_cor", "NL2_cor"],
        type=str,
        help="A list of directory names of per-target trained models.",
    )
    parser.add_argument(
        "--cut-off-value",
        default=0.1,
        type=float,
        help="Cut-off value for filtering out predictions whose confidence scores fall below the cut-off value. The cut-off value should be between 0.0 and 1.0.",
    )
    args = parser.parse_args()

    classifier = DrillCoreClassifier.from_directory(
        args.bundled_model_path,
        per_target_model_names=args.per_target_model_names,
    )
    print(classifier(args.text, cutoffval=args.cut_off_value))


if __name__ == "__main__":
    main()
