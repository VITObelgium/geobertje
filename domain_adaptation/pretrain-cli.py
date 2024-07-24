import argparse
from dataclasses import replace

from lithonlp.config import Config, get_default_config
from lithonlp.dataset import create_dataset
from lithonlp.pretrain import pretrain_model


def main() -> None:
    default_cfg = get_default_config()
    parser = argparse.ArgumentParser(
        description="Fine-tune the domain adapted model using the labeled dataset."
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
    parser.add_argument(
        "--unlabeled-dataset-file",
        default=None,
        help="File path to the input unlabeled raw dataset file (csv format).",
    )
    parser.add_argument(
        "--ignore-creating-dataset",
        default=False,
        action='store_true',
        help="Ignore creating of hugging face dataset from the input raw dataset file.",
    )
    parser.add_argument(
        "--ignore-training-model",
        default=False,
        action='store_true',
        help="Ignore training the base model (this can be used for creating only the dataset).",
    )
    parser.add_argument(
        "--test-size",
        default=default_cfg.test_size,
        type=float,
        help="Fraction of dataset to be used for testing. It must be between 0 and 1.",
    )
    parser.add_argument(
        "--sample-raw-dataset",
        default=default_cfg.sample_raw_dataset,
        type=int,
        help="Randomly sample input dataset.",
    )
    args = parser.parse_args()

    cfg = Config(
        trainer_output_dir=args.trainer_output_dir,
        pretrained_base_model_name=args.pretrained_base_model,
        pretrained_tokenizer_name=args.pretrained_tokenizer,
        trainer_num_epochs=args.epochs,
        trainer_batch_size=args.batch_size,
        test_size=args.test_size,
        class_weights_save=False,
        sample_raw_dataset=args.sample_raw_dataset,
    )

    if (args.unlabeled_dataset_file is not None) and (not args.ignore_creating_dataset):
        create_dataset(replace(cfg, raw_dataset_file=args.unlabeled_dataset_file))

    if not args.ignore_training_model:
        pretrain_model(cfg=cfg)


if __name__ == "__main__":
    main()
