import re
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert_fast import (
    BertTokenizerFast as Tokenizer,
)

from lithonlp.config import Config, get_default_config
from lithonlp.preprocesstext import preprocesstext


def create_dataset(cfg: Optional[Config] = None) -> None:
    """Create dataset per target column.

    Parameters
    ----------
    cfg : Optional[Config], optional
        input configuration, otherwise the default configuration will be used.
    """

    if cfg is None:
        cfg = get_default_config()

    df_input = pd.read_csv(cfg.raw_dataset_file, delimiter=";", low_memory=False)

    if cfg.sample_raw_dataset > 0:
        df_input = df_input.sample(cfg.sample_raw_dataset, random_state=cfg.random_seed)

    df = prepare_dataset(
        df_input, cfg.input_columns, cfg.output_columns, cfg.text_column
    )
    print(f"num_samples={len(df)}")

    df_target: pd.DataFrame = df[cfg.target_column]
    df_text: pd.DataFrame = df[cfg.text_column]

    print(f"mapping ids for '{cfg.target_column}' column")
    df[cfg.label_column] = df_target.map(cfg.label2id)

    print(f"splitting data (test_size={cfg.test_size})")
    X_train, X_test, _, _ = train_test_split(
        df_text,
        df_target,
        test_size=cfg.test_size,
        random_state=cfg.random_seed,
        stratify=df_target,
    )
    df[cfg.split_column] = "NOT_SET"
    df.loc[X_train.index, cfg.split_column] = "train"
    df.loc[X_test.index, cfg.split_column] = "test"

    if cfg.save_preprocessed_dataframe:
        filename = Path(
            cfg.raw_dataset_file.parent,
            f"preprocessed_{cfg.raw_dataset_file.name}",
        )
        df.to_csv(str(filename), sep=";", index=False)

    print("calculating class weights (to be used for model training)")
    print(f"label2id={cfg.label2id}")
    classes = list(cfg.label2id.values())
    df_train_label = df[df[cfg.split_column] == "train"][cfg.label_column]
    class_weights = compute_class_weight(
        "balanced", classes=np.array(classes), y=df_train_label
    )
    print(f"{class_weights=}")
    if cfg.class_weights_save:
        filename = Path(cfg.tokenized_hg_dataset_dir.parent, cfg.class_weights_filename)
        print(f"Saving dataset class weights into '{filename.name}'")
        np.savetxt(str(filename), class_weights.reshape(1, -1), fmt="%10.5f")

    dataset = get_hg_dataset(
        df,
        text_column=cfg.text_column,
        label_column=cfg.label_column,
        split_column=cfg.split_column,
    )
    if cfg.save_hg_dataset:
        print(f"saving hugging faces dataset (path='{cfg.hg_dataset_dir}')")
        dataset.save_to_disk(str(cfg.hg_dataset_dir))

    if cfg.save_tokenized_hg_dataset:
        print("tokenizing dataset")
        tokenizer = AutoTokenizer.from_pretrained(
            str(cfg.pretrained_tokenizer_name), do_lower_case=True
        )
        tokenized_dataset = tokenize_dataset(dataset, tokenizer)
        print(
            f"saving tokenized hugging faces dataset "
            f"(path='{cfg.tokenized_hg_dataset_dir})'"
        )
        tokenized_dataset.save_to_disk(cfg.tokenized_hg_dataset_dir)


def tokenize_dataset(dataset: Dataset, tokenizer: Tokenizer) -> Dataset:
    def tokenize(inputs: Dict[str, Any]):
        return tokenizer(
            inputs["text"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    return dataset.map(tokenize, batched=True)


def get_hg_dataset(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    split_column: str,
    splits: Tuple[str, ...] = ("train", "test"),
) -> DatasetDict:
    dataset = DatasetDict()
    for split in splits:
        df_split = df[df[split_column] == split]
        data: Dict[str, Any] = {
            "text": df_split[text_column].to_list(),
            "label": df_split[label_column].to_list(),
        }
        dataset[split] = Dataset.from_dict(data)
    return dataset


def prepare_dataset(
    dataframe: pd.DataFrame,
    input_columns: Tuple[str, ...],
    output_columns: Tuple[str, ...],
    text_column: str = "beschrijving",
    process_text: Optional[Callable[[str], str]] = None,
    fillna_value: str = "none",
) -> pd.DataFrame:
    selected_cols: List[str] = [col for col in chain(input_columns, output_columns)]
    df: pd.DataFrame = dataframe[selected_cols].copy()
    df.fillna(fillna_value, inplace=True)
    if process_text is not None:
        df[text_column].apply(process_text, inplace=True)
    return df


def select_dataset(
    dataframe: pd.DataFrame,
    output_columns: Tuple[str,],
    num_samples: int,
    minor_ratio: float,
    random_seed: int = 2023,
) -> pd.DataFrame:
    assert (
        0.0 < minor_ratio < 1.0
    ), "Unexpected minor ratio value, it must be between (0, 1)"
    num_samples_minor = int(minor_ratio * num_samples)
    num_samples_random = int((1 - minor_ratio) * num_samples)
    output_columns = list(output_columns)
    # Select top minor samples
    df_grp = (
        dataframe.drop_duplicates(subset=[*output_columns, "beschrijving"])
        .groupby(output_columns)
        .size()
        .sort_values()
    )
    select_minor = df_grp.cumsum() <= num_samples_minor
    df_minor = (
        df_grp[select_minor]
        .to_frame()
        .transform({0: lambda x: x * [1]})
        .explode(0)
        .reset_index()
    )
    # Select randomly from the rest samples
    exclude_selected_minor = df_grp.cumsum() > num_samples_minor - num_samples_random
    df_random = (
        dataframe.drop_duplicates(subset=[*output_columns, "beschrijving"])
        .reset_index()
        .set_index(output_columns)
        .loc[df_grp[exclude_selected_minor].index,]
        .sample(num_samples_random, random_state=random_seed)
    )
    # Combine the two selected dataframes
    df_minor_index = pd.MultiIndex.from_tuples(
        df_minor.groupby(output_columns).groups.keys()
    )
    df_sel = pd.concat(
        [
            dataframe.reset_index()
            .set_index(output_columns)
            .loc[df_minor_index,]
            .reset_index()
            .drop_duplicates(subset=[*output_columns, "beschrijving"]),
            df_random.reset_index(),
        ],
        ignore_index=True,
    )
    return df_sel


def preprocess_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    return s


def get_label2id(df_target: pd.DataFrame) -> Dict[str, int]:
    label2id: Dict[str, int] = {}
    labels: List[str] = list(df_target.unique())
    for index, label in enumerate(labels):
        label2id[label] = index
    return label2id
