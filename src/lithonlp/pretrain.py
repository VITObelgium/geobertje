from pathlib import Path
from typing import Optional

from datasets import load_from_disk
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from lithonlp.config import Config


def get_customized_config() -> Config:
    return Config(
        hg_dataset_dir=Path("dataset"),
        save_tokenized_hg_dataset=False,
        test_size=0.1,
        trainer_batch_size=32,
        trainer_output_dir=Path("trainer"),
    )


def pretrain_model(
    cfg: Optional[Config] = None,
) -> None:
    """Further train a pre-trained model on a large dataset of unlabeled data.

    This technique for the model adaptation is particularly useful when there is
    limited labeled data available for the target task. By leveraging the knowledge acquired
    during the pre-training on unlabeled data, the adapted model can often achieve better
    performance on the specific task than a model trained from scratch.

    Parameters
    ----------
    cfg : Optional[Config], optional
        input training configuration,
        otherwise the default configuration will be used.
    """
    if cfg is None:
        cfg = get_customized_config()

    # create_dataset(cfg)
    dataset = load_from_disk(str(cfg.hg_dataset_dir))

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained_tokenizer_name,
        do_lower_case=True,
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
            return_special_tokens_mask=True,
        )

    tokenized_dataset = dataset.map(tokenize, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["label", "text"])
    tokenized_dataset["train"]

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir=str(cfg.trainer_output_dir),
        per_device_train_batch_size=cfg.trainer_batch_size,
        per_device_eval_batch_size=cfg.trainer_batch_size,
        num_train_epochs=cfg.trainer_num_epochs,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        push_to_hub=False,
        log_level="error",
        report_to="none",
        warmup_ratio=0.05,
        load_best_model_at_end=True,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=AutoModelForMaskedLM.from_pretrained(cfg.pretrained_base_model_name),
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )

    trainer.train()
