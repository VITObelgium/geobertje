from typing import Protocol

import matplotlib.pylab as plt
import pandas as pd
from frozendict import frozendict
from scipy.stats.contingency import association
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


class Conf(Protocol):
    label2id: frozendict


def calculate_association_metric(
    df: pd.DataFrame,
    first_column: str,
    second_column: str,
) -> float:
    crosstab_results = pd.crosstab(index=df[first_column], columns=df[second_column])
    association_metric = association(crosstab_results)
    return association_metric


def get_label2id(target_column: str, default_cfg: Conf) -> frozendict:
    label2id: frozendict
    if target_column == "HL_cor":
        label2id = default_cfg.label2id
    elif (target_column == "NL1_cor") or (target_column == "NL2_cor"):
        label2id = frozendict({**default_cfg.label2id, **{"none": 9}})
    else:
        raise AssertionError(f"Unknown column target: {target_column}")
    return label2id


def plot_confusion_matrix(
    y_true,
    y_pred,
    display_labels,
    norms=("true", None),
    figsize=(10, 10),
) -> None:
    for param in norms:
        fig, ax = plt.subplots(figsize=figsize)
        ConfusionMatrixDisplay(
            confusion_matrix(y_true, y_pred, normalize=param),
            display_labels=display_labels,
        ).plot(ax=ax, cmap="gray_r", colorbar=False, xticks_rotation=45.0)
        plt.show()
