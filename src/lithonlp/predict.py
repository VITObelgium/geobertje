from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union, cast

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.models.bert.modeling_bert import (
    BertForSequenceClassification as Model,
)
from transformers.models.bert.tokenization_bert_fast import (
    BertTokenizerFast as Tokenizer,
)
from transformers.pipelines.text_classification import (
    TextClassificationPipeline as Pipeline,
)

from lithonlp.config import get_default_config


class DrillCoreClassifier:
    """A NLP-based drill core classifier that predicts multiple layers.

    It consists of multiple target column classifiers
    which all use same tokenizer but different model.
    """

    def __init__(self, tokenizer: Tokenizer, models: Dict[str, Model]) -> None:
        self.tokenizer = tokenizer
        self.models = models
        self._per_target_column_pipelines = self.build_pipelines(
            self.tokenizer, self.models
        )

    def __call__(
        self, text: Union[str, Sequence[str]], cutoffval: float = 0.0
    ) -> Dict[str, Any]:
        predictions = {
            name: pipeline(text)
            for name, pipeline in self._per_target_column_pipelines.items()
        }

        predictions = self._postprocess(predictions, cutoffval=cutoffval)

        return predictions

    @classmethod
    def from_directory(
        cls,
        path: Path,
        per_target_model_names: Tuple[str, ...] = ("HL_cor", "NL1_cor", "NL2_cor"),
        tokenizer_name: Optional[str] = None,
        best_model_name: str = "best_model",
    ) -> DrillCoreClassifier:
        base_path = Path(path)
        no_dir_error_message = "Directory '{}' does not exist!"
        if tokenizer_name is None:
            tokenizer_name = get_default_config().pretrained_tokenizer_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)
        assert base_path.is_dir(), no_dir_error_message.format(base_path)
        models: Dict[str, Model] = dict()
        for name in per_target_model_names:
            model_path = Path(base_path, name, "trainer", best_model_name)
            assert model_path.exists(), no_dir_error_message.format(model_path)
            models[name] = AutoModelForSequenceClassification.from_pretrained(
                model_path
            )
        return cls(tokenizer=cast(Tokenizer, tokenizer), models=models)

    @classmethod
    def build_pipelines(
        cls,
        tokenizer: Tokenizer,
        models: Dict,
    ) -> Dict[str, Pipeline]:
        return {
            name: cast(
                Pipeline,
                pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    top_k=None,
                ),
            )
            for name, model in models.items()
        }

    def _postprocess(
        self, predictions: Dict[str, Any], cutoffval: float
    ) -> Dict[str, Any]:
        newpredictions: Dict[str, Any] = dict()
        labels: List[str] = []
        for k, v in predictions.items():
            ds = (
                Dataset.from_list(v[0])
                .sort("score", reverse=True)
                .filter(
                    lambda row: (row["score"] >= cutoffval)
                    & (row["label"] not in labels)
                )
            )
            if ds.shape[0] > 0:
                newpredictions[k] = [ds[0]]
                labels.append(newpredictions[k][0]["label"])
            else:
                newpredictions[k] = [{"label": "none", "score": -1}]

        return newpredictions
