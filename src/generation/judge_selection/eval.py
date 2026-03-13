from generation.evaluators.deepeval_metrics import CorrectnessInput, CorrectnessResult
from generation.judge_selection.metrics import f1_macro_score, mean_absolute_error, pearson_correlation, spearman_correlation
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
    expected_ids = [row["id"] for row in input_rows]
    scores = candidate_run["scores"]
    failed_eval_ids = candidate_run["failed_eval_ids"]
    failure_reasons = candidate_run["failure_reasons"]

    score_values = list(scores.values())
    total = len(expected_ids)
    num_scored = len(score_values)
    num_failed = len(set(failed_eval_ids))
    coverage = num_scored / total if total else 0.0
    failure_rate = num_failed / total if total else 0.0

    mean_score = statistics.fmean(score_values) if score_values else None
    if len(score_values) > 1:
        score_std = statistics.pstdev(score_values)
    elif len(score_values) == 1:
        score_std = 0.0
    else:
        score_std = None

    num_compared_with_gold = 0
    pearson_value: float | None = None
    spearman_value: float | None = None
    mae_value: float | None = None
    f1_macro_value: float | None = None
    if gold_scores_by_id:
        common_ids = [
            case_id for case_id in expected_ids
            if case_id in scores and case_id in gold_scores_by_id
        ]
        num_compared_with_gold = len(common_ids)
        if common_ids:
            y_true = [float(gold_scores_by_id[case_id]) for case_id in common_ids]
            y_pred = [scores[case_id] for case_id in common_ids]
            pearson_value = pearson_correlation(y_true, y_pred)
            spearman_value = spearman_correlation(y_true, y_pred)
            mae_value = mean_absolute_error(y_true, y_pred)
            f1_macro_value = f1_macro_score(
                y_true,
                y_pred,
                score_range=score_range,
            )

    return {
        "name": candidate_run["name"],
        "judge_config": candidate_run["judge_config"],
        "coverage": coverage,
        "mean_score": mean_score,
        "score_std": score_std,
        "failure_rate": failure_rate,
        "num_scored": num_scored,
        "num_failed": num_failed,
        "num_compared_with_gold": num_compared_with_gold,
        "pearson_correlation": pearson_value,
        "spearman_correlation": spearman_value,
        "mae": mae_value,
        "f1_macro": f1_macro_value,
        "failed_eval_ids": failed_eval_ids,
        "failure_reasons": failure_reasons,
    }