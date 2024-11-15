from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from frozendict import frozendict


@dataclass
class Config:
    raw_dataset_file: Path = Path("raw_dataset.csv")
    input_columns: Tuple[str, ...] = ("beschrijving",)
    output_columns: Tuple[str, ...] = (
        "HL_cor",
        "NL1_cor",
        "NL2_cor",
    )
    text_column: str = "beschrijving"
    target_column: str = "HL_cor"
    label_column: str = ""
    split_column: str = "split"
    test_size: float = 0.15
    sample_raw_dataset: int = -1
    save_preprocessed_dataframe: bool = False
    hg_dataset_dir: Path = Path("dataset")
    save_hg_dataset: bool = True
    tokenized_hg_dataset_dir: Path = Path("tokenized_dataset")
    save_tokenized_hg_dataset: bool = True
    label2id: frozendict = frozendict(
        {
            "fijn_zand": 0,
            "grind": 1,
            "zand_onb": 2,
            "grof_zand": 3,
            "klei": 4,
            "veen": 5,
            "middel_zand": 6,
            "leem": 7,
            "silt": 8,
        }
    )
    id2label: frozendict = frozendict({})
    num_labels: int = 0
    pretrained_tokenizer_name: str = "GroNLP/bert-base-dutch-cased"
    pretrained_base_model_name: str = "GroNLP/bert-base-dutch-cased"
    trainer_output_dir: Path = Path("trainer")
    trainer_num_epochs: int = 10
    trainer_batch_size: int = 32
    trainer_learning_rate: float = 1e-4
    random_seed: int = 2023
    preprocess_text: bool = True
    class_weights_save: bool = True
    class_weights_filename: str = "class_weights.out"

    def __post_init__(self) -> None:
        self.raw_dataset_file = Path(self.raw_dataset_file)
        if not self.id2label:
            self.id2label = {v: k for k, v in self.label2id.items()}
        if self.num_labels == 0:
            self.num_labels = len(self.label2id)
        if not self.label_column:
            self.label_column = f"label [{self.target_column}]"


def get_default_config() -> Config:
    """Get a new instance of the default configuration."""
    return Config()
