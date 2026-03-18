from generation.evaluators.deepeval_metrics import CorrectnessInput, CorrectnessResult
from generation.judge_selection.metrics import (
    f1_macro_score_with_per_label,
    group_labels,
    mean_absolute_error,
    pearson_correlation,
    spearman_correlation,
)
from generation.judge_selection.schema import CandidateCorrectnessRun, JudgeCandidate, JudgeEvaluation, JudgeConfig
from typing import Callable
import statistics

EvaluatorFn = Callable[[list[CorrectnessInput], JudgeConfig | None], CorrectnessResult]

def _normalize_candidate_correctness_run(
    candidate: JudgeCandidate,
    input_rows: list[CorrectnessInput],
    raw_result: CorrectnessResult,
    *,
    fill_missing_failures: bool = True,
) -> CandidateCorrectnessRun:
    expected_ids = [row["id"] for row in input_rows]
    expected_set = set(expected_ids)

    raw_scores = raw_result.get("scores", {})
    scores: dict[str, float] = {}
    if isinstance(raw_scores, dict):
        for case_id, score in raw_scores.items():
            normalized_id = str(case_id).strip()
            if not normalized_id or normalized_id not in expected_set:
                continue
            try:
                scores[normalized_id] = float(score)
            except (TypeError, ValueError):
                continue

    failed_eval_ids: list[str] = []
    seen_failed_ids: set[str] = set()
    raw_failed_eval_ids = raw_result.get("failed_eval_ids", [])
    if isinstance(raw_failed_eval_ids, list):
        for case_id in raw_failed_eval_ids:
            normalized_id = str(case_id).strip()
            if (
                not normalized_id
                or normalized_id not in expected_set
                or normalized_id in seen_failed_ids
            ):
                continue
            seen_failed_ids.add(normalized_id)
            failed_eval_ids.append(normalized_id)

    failure_reasons: dict[str, str] = {}
    raw_failure_reasons = raw_result.get("failure_reasons", {})
    if isinstance(raw_failure_reasons, dict):
        for case_id, reason in raw_failure_reasons.items():
            normalized_id = str(case_id).strip()
            if not normalized_id or normalized_id not in expected_set:
                continue
            failure_reasons[normalized_id] = str(reason)

    evaluation_reasons: dict[str, str] = {}
    raw_evaluation_reasons = raw_result.get("evaluation_reasons", {})
    if isinstance(raw_evaluation_reasons, dict):
        for case_id, reason in raw_evaluation_reasons.items():
            normalized_id = str(case_id).strip()
            if not normalized_id or normalized_id not in expected_set:
                continue
            evaluation_reasons[normalized_id] = str(reason)

    if fill_missing_failures:
        scored_ids = set(scores.keys())
        failed_ids = set(failed_eval_ids)
        for case_id in expected_ids:
            if case_id in scored_ids or case_id in failed_ids:
                continue
            failed_eval_ids.append(case_id)
            failure_reasons[case_id] = "missing_test_result"

        for case_id in failed_eval_ids:
            failure_reasons.setdefault(case_id, "missing_test_result")

    return {
        "name": candidate["name"],
        "judge_config": candidate["judge_config"],
        "scores": scores,
        "failed_eval_ids": failed_eval_ids,
        "failure_reasons": failure_reasons,
        "evaluation_reasons": evaluation_reasons,
    }


def _evaluate_candidate_correctness(
    candidate: JudgeCandidate,
    input_rows: list[CorrectnessInput],
    evaluator: EvaluatorFn,
) -> CandidateCorrectnessRun:
    expected_ids = [row["id"] for row in input_rows]
    try:
        raw_result = evaluator(input_rows, judge_config=candidate["judge_config"])
    except Exception as exc:
        raw_result = {
            "scores": {},
            "failed_eval_ids": expected_ids,
            "failure_reasons": {case_id: str(exc) for case_id in expected_ids},
            "evaluation_reasons": {},
        }
    return _normalize_candidate_correctness_run(candidate, input_rows, raw_result)


def _merge_candidate_runs(
    candidate: JudgeCandidate,
    input_rows: list[CorrectnessInput],
    cached_run: CandidateCorrectnessRun,
    new_run: CandidateCorrectnessRun,
) -> CandidateCorrectnessRun:
    merged_scores: dict[str, float] = {}
    merged_failed_eval_ids: set[str] = set()
    merged_failure_reasons: dict[str, str] = {}
    merged_evaluation_reasons: dict[str, str] = {}

    merged_scores.update(cached_run["scores"])
    merged_failed_eval_ids.update(cached_run["failed_eval_ids"])
    merged_failure_reasons.update(cached_run["failure_reasons"])
    merged_evaluation_reasons.update(cached_run["evaluation_reasons"])

    merged_scores.update(new_run["scores"])
    merged_failed_eval_ids.update(new_run["failed_eval_ids"])
    merged_failure_reasons.update(new_run["failure_reasons"])
    merged_evaluation_reasons.update(new_run["evaluation_reasons"])

    # If a case now has a score, it should no longer be marked as failed.
    scored_ids = set(merged_scores.keys())
    merged_failed_eval_ids -= scored_ids
    for case_id in scored_ids:
        merged_failure_reasons.pop(case_id, None)

    merged_raw: CorrectnessResult = {
        "scores": merged_scores,
        "failed_eval_ids": sorted(merged_failed_eval_ids),
        "failure_reasons": merged_failure_reasons,
        "evaluation_reasons": merged_evaluation_reasons,
    }
    return _normalize_candidate_correctness_run(
        candidate,
        input_rows,
        merged_raw,
        fill_missing_failures=True,
    )


def _compute_judge_evaluation(
    candidate_run: CandidateCorrectnessRun,
    input_rows: list[CorrectnessInput],
    score_range: tuple[float, float],
    gold_scores_by_id: dict[str, float] | None,
) -> JudgeEvaluation:
    outputs = {}
    expected_ids = [row["id"] for row in input_rows]
    scores = candidate_run["scores"]
    outputs["failed_eval_ids"] = candidate_run["failed_eval_ids"]
    outputs["failure_reasons"] = candidate_run["failure_reasons"]

    score_values = list(scores.values())
    total = len(expected_ids)
    outputs["num_scored"] = len(score_values)
    outputs["num_failed"] = len(set(outputs["failed_eval_ids"]))

    outputs["coverage"] = outputs["num_scored"] / total if total else 0.0
    outputs["failure_rate"] = outputs["num_failed"] / total if total else 0.0

    outputs["mean_score"] = statistics.fmean(score_values) if score_values else None
    if len(score_values) > 1:
        outputs["score_std"] = statistics.pstdev(score_values)
    elif len(score_values) == 1:
        outputs["score_std"]  = 0.0
    else:
        outputs["score_std"]  = None

    outputs["num_compared_with_gold"] = 0
    if gold_scores_by_id:
        common_ids = [
            case_id for case_id in expected_ids
            if case_id in scores and case_id in gold_scores_by_id
        ]
        outputs["num_compared_with_gold"] = len(common_ids)
        if common_ids:
            y_true = [float(gold_scores_by_id[case_id]) for case_id in common_ids]
            y_pred = [scores[case_id] for case_id in common_ids]
            outputs["mean_true_score"] = statistics.fmean(y_true)
            low_label = int(score_range[0])
            high_label = int(score_range[1])
            pred_values_by_true_label: dict[str, list[float]] = {}
            for true_value, pred_value in zip(y_true, y_pred):
                true_label = int(round(true_value))
                true_label = max(low_label, min(high_label, true_label))
                key = str(true_label)
                pred_values_by_true_label.setdefault(key, []).append(float(pred_value))
            outputs["mean_pred_by_true_label"] = {
                label: statistics.fmean(values)
                for label, values in pred_values_by_true_label.items()
            }
            outputs["num_by_true_label"] = {
                label: len(values)
                for label, values in pred_values_by_true_label.items()
            }
            outputs["pearson"] = pearson_correlation(y_true, y_pred)
            outputs["spearman"]  = spearman_correlation(y_true, y_pred)
            outputs["mae"] = mean_absolute_error(y_true, y_pred)
            outputs["f1_macro"], outputs["f1_macro_per_label"] = f1_macro_score_with_per_label(
                y_true,
                y_pred,
                score_range=score_range,
            )


            outputs["f1_macro_t_3"], outputs["f1_macro_per_label_t_3"] = f1_macro_score_with_per_label(
                group_labels(y_true, groups={"1-2": (1, 2), "3-5": (3, 4, 5)}),
                group_labels(y_pred, groups={"1-2": (1, 2), "3-5": (3, 4, 5)}),
            )

            outputs["f1_macro_t_4"], outputs["f1_macro_per_label_t_4"] = f1_macro_score_with_per_label(
                group_labels(y_true, groups={"1-3": (1, 2, 3), "4-5": (4, 5)}),
                group_labels(y_pred, groups={"1-3": (1, 2, 3), "4-5": (4, 5)}),
            )

            outputs["f1_macro_group_1_2_vs_3_vs_4_5"], outputs[
                "f1_macro_per_label_group_1_2_vs_3_vs_4_5"
            ] = (
                f1_macro_score_with_per_label(
                    group_labels(y_true, groups={"1-2": (1, 2), "3": (3,), "4-5": (4, 5)}),
                    group_labels(y_pred, groups={"1-2": (1, 2), "3": (3,), "4-5": (4, 5)}),
                )
            )



    return outputs
