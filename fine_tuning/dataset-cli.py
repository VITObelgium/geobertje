import argparse

import pandas as pd
from datasets import load_from_disk

from lithonlp import utils
from lithonlp.config import Config, get_default_config
from lithonlp.dataset import create_dataset


def main() -> None:
    default_cfg = get_default_config()
    parser = argparse.ArgumentParser(
        description="Prepare dataset for the NLP-based drill core classification."
    )
    parser.add_argument(
        "-f",
        "--raw-dataset-file",
        default=default_cfg.raw_dataset_file,
        type=str,
        help="raw dataset file name.",
    )
    parser.add_argument(
        "-s",
        "--sample-raw-dataset",
        default=default_cfg.sample_raw_dataset,
        type=int,
        help="number of randomly selected samples from the raw dataset.",
    )
    parser.add_argument(
        "-t",
        "--target-column",
        default=default_cfg.target_column,
        help="target column: either of HL_cor, NL1_cor, or NL2_cor.",
    )
    parser.add_argument(
        "--export-csv",
        default=None,
        help="folder path where train/test datasets are saved into separate csv files.",
    )
    parser.add_argument(
        "--test-size",
        default=default_cfg.test_size,
        type=float,
        help="Fraction of dataset to be used for testing. It must be between 0 and 1.",
    )
    args = parser.parse_args()
    cfg = Config(
        raw_dataset_file=args.raw_dataset_file,
        sample_raw_dataset=args.sample_raw_dataset,
        target_column=args.target_column,
        label2id=utils.get_label2id(args.target_column, default_cfg),
        test_size=args.test_size,
    )
    create_dataset(cfg)

    if args.export_csv is not None:
        dataset = load_from_disk(cfg.hg_dataset_dir)
        for split in ("train", "test"):
            df: pd.DataFrame = dataset[split].to_pandas()
            df["label"] = df["label"].apply(lambda v: cfg.id2label[v])
            df.to_csv(rf"{args.export_csv}//{split}set_{args.target_column}.csv")


if __name__ == "__main__":
    main()
