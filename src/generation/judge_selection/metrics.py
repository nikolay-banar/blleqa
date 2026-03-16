import math
import statistics
from typing import Hashable

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _to_discrete_label(value: float, score_range: tuple[float, float]) -> int:
    low = int(math.ceil(score_range[0]))
    high = int(math.floor(score_range[1]))
    if high < low:
        return int(round(value))
    return int(_clamp(round(value), low, high))


def pearson_correlation(y_true: list[float], y_pred: list[float]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("y_true and y_pred must be non-empty")
    result = pearsonr(y_true, y_pred)
    corr = float(result.statistic)
    return 0.0 if math.isnan(corr) else corr


def spearman_correlation(y_true: list[float], y_pred: list[float]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("y_true and y_pred must be non-empty")
    result = spearmanr(y_true, y_pred)
    corr = float(result.statistic)
    return 0.0 if math.isnan(corr) else corr


def mean_absolute_error(y_true: list[float], y_pred: list[float]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("y_true and y_pred must be non-empty")
    return statistics.fmean(abs(true_value - pred_value) for true_value, pred_value in zip(y_true, y_pred))


def to_binary_labels(values: list[float], *, threshold: float = 3.0) -> list[int]:
    return [0 if float(value) < threshold else 1 for value in values]


def accuracy(y_true: list[Hashable], y_pred: list[Hashable]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("y_true and y_pred must be non-empty")
    return float(accuracy_score(y_true, y_pred))


def f1_binary_score(y_true: list[int], y_pred: list[int]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("y_true and y_pred must be non-empty")
    return float(f1_score(y_true, y_pred, average="binary", zero_division=0))


def f1_macro_score(
    y_true: list[Hashable],
    y_pred: list[Hashable],
    *,
    score_range: tuple[float, float] | None = None,
) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("y_true and y_pred must be non-empty")

    if score_range is not None:
        y_true_labels = [_to_discrete_label(float(value), score_range) for value in y_true]
        y_pred_labels = [_to_discrete_label(float(value), score_range) for value in y_pred]
    else:
        y_true_labels = list(y_true)
        y_pred_labels = list(y_pred)

    labels = sorted(set(y_true_labels) | set(y_pred_labels), key=lambda item: str(item))
    return float(
        f1_score(
            y_true_labels,
            y_pred_labels,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


__all__ = [
    "pearson_correlation",
    "spearman_correlation",
    "mean_absolute_error",
    "to_binary_labels",
    "accuracy",
    "f1_binary_score",
    "f1_macro_score",
]
