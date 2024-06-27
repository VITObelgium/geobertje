import numpy as np
import pandas as pd
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from lithonlp.config import get_default_config
from lithonlp.predict import DrillCoreClassifier
from lithonlp.utils import get_label2id
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef

disable_progress_bar()

if __name__ == "__main__":
    GEOBERTJE_PATH: str = "/projects/y-drive/un_rma/_GEO/Projecten/GEOBertje/6_Trained_models/fine_tuned/bundled/best"

    default_cfg = get_default_config()
    label2id = get_label2id("NL1_cor", default_cfg)

    classifier = DrillCoreClassifier.from_directory(
        path=GEOBERTJE_PATH,
        per_target_model_names=("HL_cor", "NL1_cor", "NL2_cor"),
    )

    cutoffvals: np.array = np.arange(0.1, 0.7, 0.05)
    accuracies: pd.DataFrame = pd.DataFrame(
        index=pd.RangeIndex(len(cutoffvals)), columns=["CutOffVal", "HL", "NL1", "NL2"]
    )
    accuracies.loc[:, "CutOffVal"] = cutoffvals
    for target in ("HL", "NL1", "NL2"):
        testds: Dataset = Dataset.load_from_disk(
            f"{GEOBERTJE_PATH}/{target}_cor/dataset/test"
        )
        for ind, cutoffval in enumerate(cutoffvals):
            res = testds.map(
                lambda row: {
                    "pred": label2id[
                        classifier(row["text"], cutoffval=cutoffval)[f"{target}_cor"][
                            0
                        ]["label"]
                    ]
                }
            )
            accuracies.loc[ind, target] = accuracy_score(
                y_true=res["label"], y_pred=res["pred"]
            )

    print(accuracies)
    print(accuracies.set_index("CutOffVal").mean(axis=1))
    print(
        f"Optimal cut-off value is {accuracies.set_index('CutOffVal').mean(axis=1).convert_dtypes().idxmax()}"
    )
    # Optimal cut-off value for different metrics:
    # accuracy score: 0.1
    # balanced accuracy score: 0.1
    # matthews_corrcoef: 0.1
