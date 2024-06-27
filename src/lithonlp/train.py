from typing import Optional

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from lithonlp.config import Config, get_default_config


def train_model(
    cfg: Optional[Config] = None,
    class_weights: Optional[np.ndarray] = None,
) -> None:
    """Fine tune the base pretrained model with an attached classifier.

    Parameters
    ----------
    cfg : Optional[Config], optional
        input training configuration,
        otherwise the default configuration will be used.
    class_weights : Optional[np.ndarray], optional
        class weights to deal with imbalanced dataset,
        by default hugging face trainer uses no class weights.
    """
    if cfg is None:
        cfg = get_default_config()

    tokenized_dataset = load_from_disk(str(cfg.tokenized_hg_dataset_dir))
    train_dataset = tokenized_dataset["train"].shuffle()
    eval_dataset = tokenized_dataset["test"].shuffle()
    print("train dataset samples:", len(tokenized_dataset["train"]))
    print("test dataset samples:", len(tokenized_dataset["test"]))

    print(f"base model: {cfg.pretrained_base_model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.pretrained_base_model_name,
        num_labels=cfg.num_labels,
        label2id=cfg.label2id,
        id2label=cfg.id2label,
    )

    metric = evaluate.load("accuracy")
    # metric = evaluate.combine(["accuracy", "f1"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    Trainer_: type[Trainer]
    if class_weights is not None:
        assert (
            len(class_weights) == cfg.num_labels
        ), "Unexpected length for class weights array"

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        class_weights_tensor = torch.tensor(
            class_weights, device=device, dtype=model.dtype
        )

        # https://discuss.huggingface.co/t/how-can-i-use-class-weights-when-training/1067/7
        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
                loss = loss_fct(
                    logits.view(-1, self.model.config.num_labels),
                    labels.view(-1),
                )
                return (loss, outputs) if return_outputs else loss

        print(f"Using custom trainer ({class_weights=})")
        Trainer_ = CustomTrainer
    else:
        print("Using default trainer")
        Trainer_ = Trainer

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
        warmup_ratio=0.20,
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
    )

    trainer = Trainer_(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
