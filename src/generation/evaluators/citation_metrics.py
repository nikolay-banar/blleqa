import statistics
from typing import Any


def citation_score(y_true: list[list[str]], y_pred: list[list[str]]) -> tuple[list[float], list[float], list[float]]:
  assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
  precision = []
  recall = []
  f1 = []
  for i_true, i_pred in zip(y_true, y_pred):
    i_pred = set(i_pred)
    i_true = set(i_true)

    n_pred = len(i_pred)
    n_true = len(i_true)

    if n_pred == 0 and n_true == 0:
      precision.append(1.0); recall.append(1.0); f1.append(1.0)
      continue

    tp = len(i_pred & i_true)


    i_precision = tp/n_pred if n_pred != 0 else 0.0
    i_recall = tp/n_true if n_true != 0 else 0.0
    denom = i_precision + i_recall

    i_f1 = 2 * i_precision * i_recall / denom if denom != 0 else 0.0

    precision.append(i_precision)
    recall.append(i_recall)
    f1.append(i_f1)
  return precision, recall, f1


def _evaluate_citations(
    *,
    gold_by_qid: dict[str, list[str]],
    predicted_by_qid: dict[str, list[str]],
    refusals_by_qid: dict[str, bool],
    expected_refusal_by_qid: dict[str, bool],
) -> dict[str, Any]:
    gold_ids = set(gold_by_qid.keys())
    predicted_ids = set(predicted_by_qid.keys())
    if gold_ids != predicted_ids:
        raise ValueError(
            "gold_by_qid and predicted_by_qid must contain the same ids; "
            f"id_diff={sorted(gold_ids ^ predicted_ids)}"
        )

    candidate_ids = sorted(gold_ids)
    unfiltered_y_true = [gold_by_qid[qid] for qid in candidate_ids]
    unfiltered_y_pred = [predicted_by_qid[qid] for qid in candidate_ids]

    (
        unfiltered_precision_values,
        unfiltered_recall_values,
        unfiltered_f1_values,
    ) = citation_score(unfiltered_y_true, unfiltered_y_pred)
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []

    by_id: dict[str, dict[str, Any]] = {}
    for (
        qid,
        raw_gold_ids,
        raw_predicted_ids,
        unfiltered_precision,
        unfiltered_recall,
        unfiltered_f1,
    ) in zip(
        candidate_ids,
        unfiltered_y_true,
        unfiltered_y_pred,
        unfiltered_precision_values,
        unfiltered_recall_values,
        unfiltered_f1_values,
    ):  
        expected_refusal = bool(expected_refusal_by_qid.get(qid, False))
        predicted_refusal = bool(refusals_by_qid.get(qid, False))

        if expected_refusal and predicted_refusal:
           precision = 1.0
           recall = 1.0
           f1 = 1.0
        elif not expected_refusal and predicted_refusal:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            precision = unfiltered_precision
            recall = unfiltered_recall
            f1 = unfiltered_f1
        
           
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

        by_id[qid] = {
            "expected_refusal": expected_refusal,
            "predicted_refusal": predicted_refusal,
            "unfiltered_precision": unfiltered_precision,
            "unfiltered_recall": unfiltered_recall,
            "unfiltered_f1": unfiltered_f1,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "raw_gold_ids": raw_gold_ids,
            "raw_predicted_ids": raw_predicted_ids,
        }

    precision_mean = statistics.fmean(precision_values) if precision_values else None
    recall_mean = statistics.fmean(recall_values) if recall_values else None
    f1_mean = statistics.fmean(f1_values) if f1_values else None
    unfiltered_precision_mean = (
        statistics.fmean(unfiltered_precision_values) if unfiltered_precision_values else None
    )
    unfiltered_recall_mean = (
        statistics.fmean(unfiltered_recall_values) if unfiltered_recall_values else None
    )
    unfiltered_f1_mean = statistics.fmean(unfiltered_f1_values) if unfiltered_f1_values else None

    return {
        "metrics": {
            "unfiltered_precision_mean": unfiltered_precision_mean,
            "unfiltered_recall_mean": unfiltered_recall_mean,
            "unfiltered_f1_mean": unfiltered_f1_mean,
            "precision_mean": precision_mean,
            "recall_mean": recall_mean,
            "f1_mean": f1_mean,
        },
        "counts": {
            "num_gold_rows": len(gold_by_qid),
            "num_predicted_rows": len(predicted_by_qid),
            "num_compared_rows": len(candidate_ids),
            "num_missing_from_generation": 0,
            "num_extra_in_generation": 0,
        },
        "missing_from_generation": [],
        "extra_in_generation": [],
        "by_id": by_id,
    }
