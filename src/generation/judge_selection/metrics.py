import math
import statistics
from typing import Hashable, Iterable

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


def group_labels(
    values: list[float],
    *,
    threshold: float = 3.0,
    groups: (
        dict[str, list[int] | tuple[int, ...] | set[int]]
        | list[list[int] | tuple[int, ...] | set[int]]
        | None
    ) = None,
    score_range: tuple[float, float] = (0.0, 5.0),
    fallback_group: Hashable | None = None,
) -> list[Hashable]:
    # Default behavior remains binary thresholding for backward compatibility.
    if groups is None:
        return [0 if float(value) < threshold else 1 for value in values]

    def _register_group(
        target: dict[int, Hashable],
        group_key: Hashable,
        group_labels: Iterable[int],
    ) -> None:
        for label in group_labels:
            normalized_label = int(label)
            if normalized_label in target:
                raise ValueError(f"Label {normalized_label} appears in multiple groups.")
            target[normalized_label] = group_key

    label_to_group: dict[int, Hashable] = {}
    if isinstance(groups, dict):
        for group_name, group in groups.items():
            _register_group(label_to_group, str(group_name), group)
    else:
        for group_index, group in enumerate(groups):
            _register_group(label_to_group, group_index, group)

    grouped_labels: list[Hashable] = []
    for value in values:
        label = _to_discrete_label(float(value), score_range)
        if label in label_to_group:
            grouped_labels.append(label_to_group[label])
            continue
        if fallback_group is not None:
            grouped_labels.append(fallback_group)
            continue
        raise ValueError(
            f"Label {label} is not present in any configured group "
            f"(groups={groups}, score_range={score_range})."
        )
    return grouped_labels


def accuracy(y_true: list[Hashable], y_pred: list[Hashable]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("y_true and y_pred must be non-empty")
    return float(accuracy_score(y_true, y_pred))


def f1_macro_score(
    y_true: list[Hashable],
    y_pred: list[Hashable],
    *,
    score_range: tuple[float, float] | None = None,
) -> float:
    score, _ = f1_macro_score_with_per_label(
        y_true,
        y_pred,
        score_range=score_range,
    )
    return score


def f1_macro_score_with_per_label(
    y_true: list[Hashable],
    y_pred: list[Hashable],
    *,
    score_range: tuple[float, float] | None = None,
) -> tuple[float, dict[str, float]]:
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
    per_label_scores = f1_score(
        y_true_labels,
        y_pred_labels,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_label_score_map = {
        str(label): float(score)
        for label, score in zip(labels, per_label_scores)
    }
    return statistics.fmean(per_label_score_map.values()), per_label_score_map


__all__ = [
    "pearson_correlation",
    "spearman_correlation",
    "mean_absolute_error",
    "group_labels",
    "accuracy",
    "f1_macro_score",
    "f1_macro_score_with_per_label",
]
