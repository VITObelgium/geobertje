import argparse
from typing import Optional

import numpy as np

from lithonlp import utils
from lithonlp.config import Config, get_default_config
from lithonlp.train import train_model


def main() -> None:
    default_cfg = get_default_config()
    parser = argparse.ArgumentParser(
        description="Fine-tune hugging face model on provided drill core dataset."
    )
    parser.add_argument(
        "--class-weights",
        nargs="+",
        default=None,
        type=float,
        help="class weights used for model training, default to None.",
    )
    parser.add_argument(
        "--target-column",
        default=default_cfg.target_column,
        help="target column: either of HL_cor, NL1_cor, or NL2_cor.",
    )
    parser.add_argument(
        "--trainer-output-dir",
        default=default_cfg.trainer_output_dir,
        help="Path to the trainer output directory",
    )
    parser.add_argument(
        "--pretrained-base-model",
        default=default_cfg.pretrained_base_model_name,
        help="Name or path to the pretrained base model",
    )
    parser.add_argument(
        "--pretrained-tokenizer",
        default=default_cfg.pretrained_tokenizer_name,
        help="Name or path to the pretrained tokenizer.",
    )
    parser.add_argument(
        "--tokenized-hg-dataset-dir",
        default=default_cfg.tokenized_hg_dataset_dir,
        help="Path to the input hugging face tokenized dataset.",
    )
    parser.add_argument(
        "--epochs",
        default=default_cfg.trainer_num_epochs,
        type=int,
        help="number of epochs when training model.",
    )
    parser.add_argument(
        "--batch-size",
        default=default_cfg.trainer_batch_size,
        type=int,
        help="number of samples in each batch.",
    )
    args = parser.parse_args()

    cfg = Config(
        target_column=args.target_column,
        trainer_num_epochs=args.epochs,
        trainer_batch_size=args.batch_size,
        label2id=utils.get_label2id(args.target_column, default_cfg),
        pretrained_base_model_name=args.pretrained_base_model,
        pretrained_tokenizer_name=args.pretrained_tokenizer,
        trainer_output_dir=args.trainer_output_dir,
        tokenized_hg_dataset_dir=args.tokenized_hg_dataset_dir,
    )

    class_weights: Optional[np.ndarray]
    if args.class_weights is None:
        class_weights = None
    else:
        assert len(args.class_weights) == len(cfg.label2id)
        class_weights = args.class_weights

    train_model(cfg=cfg, class_weights=class_weights)


if __name__ == "__main__":
    main()
