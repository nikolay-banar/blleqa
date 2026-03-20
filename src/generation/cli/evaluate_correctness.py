from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

from generation.cli.evaluate_citations import _load_prediction_text_by_qid
from generation.cli.evaluate_refusals import (
    DEFAULT_DATASET_ID,
    DEFAULT_SPLIT,
    _default_output_dir,
    _derive_expected_refusal_by_qid,
    _load_context_data_by_qid,
    _load_generation_answers,
    _load_refusal_flags_by_qid,
    _print_rows,
    _refusal_case_label,
)
from generation.evaluators import deepeval_correctness
from generation.pipeline.bbleqa import _load_gold_ids_by_qid, _load_gold_query_ref_by_qid

DEFAULT_JUDGE_MODEL = "gemini-3-flash"
BINARY_THRESHOLD = 4.0
CORRECTNESS_TABLE_COLUMNS = [
    "Model",
    "Setup",
    "Cov",
    "ScoreMean",
    "ScoreMeanCorrected",
    "BinAcc",
    "BinAccCorrected",
    "CorrectAnswers",
    "NonCorrect",
    "CorrectRefusals",
    "IncorrectRefusals",
]

OUTCOME_KEYS = [
    "correct_answers",
    "non_correct",
    "correct_refusals",
    "incorrect_refusals",
]


def _build_deepeval_rows(
    *,
    query_by_qid: dict[str, str],
    ref_by_qid: dict[str, str],
    prediction_text_by_qid: dict[str, str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    candidate_ids = sorted(
        set(query_by_qid.keys()) & set(ref_by_qid.keys()) & set(prediction_text_by_qid.keys())
    )
    for qid in candidate_ids:
        query = query_by_qid.get(qid, "").strip()
        ref = ref_by_qid.get(qid, "").strip()
        prediction = prediction_text_by_qid.get(qid, "").strip()
        if not query or not ref or not prediction:
            continue
        rows.append(
            {
                "id": qid,
                "query": query,
                "prediction": prediction,
                "ref": ref,
            }
        )
    return rows


def _compute_outcome_breakdown(
    *,
    row_ids: list[str],
    scores_by_id: dict[str, float],
    expected_refusal_by_qid: dict[str, bool] | None,
    refusals_by_qid: dict[str, bool] | None,
    binary_threshold: float = BINARY_THRESHOLD,
) -> dict[str, object] | None:
    if expected_refusal_by_qid is None or refusals_by_qid is None:
        return None

    counts = {key: 0 for key in OUTCOME_KEYS}
    total = len(row_ids)

    for case_id in row_ids:
        category = _resolve_outcome_category(
            case_id=case_id,
            scores_by_id=scores_by_id,
            expected_refusal_by_qid=expected_refusal_by_qid,
            refusals_by_qid=refusals_by_qid,
            binary_threshold=binary_threshold,
        )
        if category is None:
            continue
        counts[category] += 1

    fractions = {
        key: (counts[key] / total if total else None)
        for key in OUTCOME_KEYS
    }
    return {
        "counts": counts,
        "fractions": fractions,
        "total": total,
        "binary_threshold": binary_threshold,
    }


def _resolve_outcome_category(
    *,
    case_id: str,
    scores_by_id: dict[str, float],
    expected_refusal_by_qid: dict[str, bool] | None,
    refusals_by_qid: dict[str, bool] | None,
    binary_threshold: float = BINARY_THRESHOLD,
) -> str | None:
    if expected_refusal_by_qid is None or refusals_by_qid is None:
        return None
    expected_refusal = bool(expected_refusal_by_qid.get(case_id, False))
    predicted_refusal = bool(refusals_by_qid.get(case_id, False))
    if predicted_refusal:
        return "correct_refusals" if expected_refusal else "incorrect_refusals"
    
    score = scores_by_id.get(case_id)
    return "correct_answers" if score is not None and float(score) >= binary_threshold else "non_correct"


def _build_correctness_by_id(
    *,
    row_ids: list[str],
    raw_scores_by_id: dict[str, float],
    corrected_scores_by_id: dict[str, float],
    refusal_score_overrides: dict[str, float],
    failed_eval_ids: list[str],
    failure_reasons: dict[str, str],
    evaluation_reasons: dict[str, str],
    expected_refusal_by_qid: dict[str, bool] | None,
    refusals_by_qid: dict[str, bool] | None,
    binary_threshold: float = BINARY_THRESHOLD,
) -> dict[str, dict[str, object]]:
    by_id: dict[str, dict[str, object]] = {}
    failed_set = set(failed_eval_ids)

    for case_id in row_ids:
        expected_refusal = (
            bool(expected_refusal_by_qid.get(case_id, False))
            if expected_refusal_by_qid is not None
            else None
        )
        predicted_refusal = (
            bool(refusals_by_qid.get(case_id, False))
            if refusals_by_qid is not None
            else None
        )
        by_id[case_id] = {
            "score": raw_scores_by_id.get(case_id),
            "corrected_score": corrected_scores_by_id.get(case_id),
            "refusal_score_override": refusal_score_overrides.get(case_id),
            "expected_refusal": expected_refusal,
            "predicted_refusal": predicted_refusal,
            "refusal_case": _refusal_case_label(
                expected_refusal=expected_refusal,
                predicted_refusal=predicted_refusal,
            ),
            "outcome_before_refusal_correction": _resolve_outcome_category(
                case_id=case_id,
                scores_by_id=raw_scores_by_id,
                expected_refusal_by_qid=expected_refusal_by_qid,
                refusals_by_qid=refusals_by_qid,
                binary_threshold=binary_threshold,
            ),
            "outcome_after_refusal_correction": _resolve_outcome_category(
                case_id=case_id,
                scores_by_id=corrected_scores_by_id,
                expected_refusal_by_qid=expected_refusal_by_qid,
                refusals_by_qid=refusals_by_qid,
                binary_threshold=binary_threshold,
            ),
            "failed_eval": case_id in failed_set,
            "failure_reason": failure_reasons.get(case_id),
            "evaluation_reason": evaluation_reasons.get(case_id),
        }
    return by_id


def run_correctness_evaluation(
    *,
    query_by_qid: dict[str, str],
    ref_by_qid: dict[str, str],
    prediction_text_by_qid: dict[str, str],
    judge_model: str,
    cached_payload: dict[str, object] | None,
    expected_refusal_by_qid: dict[str, bool] | None = None,
    refusals_by_qid: dict[str, bool] | None = None,
) -> dict[str, Any]:
    from generation.judge_selection.eval import (
        _evaluate_candidate_correctness,
        _merge_candidate_runs,
        _normalize_candidate_correctness_run,
    )

    all_rows = _build_deepeval_rows(
        query_by_qid=query_by_qid,
        ref_by_qid=ref_by_qid,
        prediction_text_by_qid=prediction_text_by_qid,
    )
    deepeval_rows: list[dict[str, str]] = all_rows

    candidate = {
        "name": judge_model,
        "judge_config": {"model": judge_model},
    }
    cached_raw: object = {}
    if isinstance(cached_payload, dict):
        cached_raw = cached_payload.get("deepeval", {})
    cached_run = _normalize_candidate_correctness_run(
        candidate,
        deepeval_rows,
        cached_raw if isinstance(cached_raw, dict) else {},
        fill_missing_failures=False,
    )
    processed_ids = set(cached_run["scores"].keys())
    rows_to_evaluate = [
        row for row in deepeval_rows if str(row["id"]) not in processed_ids
    ]
    if processed_ids:
        print(
            "DeepEval cache: "
            f"reusing {len(processed_ids)}/{len(deepeval_rows)} scored rows; "
            f"evaluating {len(rows_to_evaluate)} missing rows."
        )

    new_run = {
        "name": candidate["name"],
        "judge_config": candidate["judge_config"],
        "scores": {},
        "failed_eval_ids": [],
        "failure_reasons": {},
        "evaluation_reasons": {},
    }
    if rows_to_evaluate:
        new_run = _evaluate_candidate_correctness(
            candidate,
            rows_to_evaluate,
            deepeval_correctness,
        )
    merged_run = _merge_candidate_runs(
        candidate,
        deepeval_rows,
        cached_run,
        new_run,
    )

    forced_scores: dict[str, float] = {}
    if expected_refusal_by_qid is not None and refusals_by_qid is not None:
        for row in all_rows:
            case_id = str(row["id"])
            expected_refusal = bool(expected_refusal_by_qid.get(case_id, False))
            predicted_refusal = bool(refusals_by_qid.get(case_id, False))
            if predicted_refusal:
                if expected_refusal:
                    forced_scores[case_id] = 5.0
                else:
                    forced_scores[case_id] = 1.0
    if forced_scores:
        print(
            "Correctness refusal override: "
            f"forced_scores={len(forced_scores)} "
            f"(correct refusal -> 5.0, incorrect refusal -> 1.0)."
        )
    raw_scores_by_id = dict(merged_run["scores"])
    corrected_scores = dict(raw_scores_by_id)
    corrected_scores.update(forced_scores)
    failed_eval_ids = [
        case_id for case_id in merged_run["failed_eval_ids"] if case_id not in forced_scores
    ]
    failure_reasons = {
        case_id: reason
        for case_id, reason in merged_run["failure_reasons"].items()
        if case_id not in forced_scores
    }
    evaluation_reasons = {
        case_id: reason
        for case_id, reason in merged_run["evaluation_reasons"].items()
        if case_id not in forced_scores
    }
    result = {
        "scores": raw_scores_by_id,
        "corrected_scores": corrected_scores,
        "refusal_score_overrides": forced_scores,
        "failed_eval_ids": failed_eval_ids,
        "failure_reasons": failure_reasons,
        "evaluation_reasons": evaluation_reasons,
    }

    row_ids = [str(row["id"]) for row in all_rows]
    result["by_id"] = _build_correctness_by_id(
        row_ids=row_ids,
        raw_scores_by_id=raw_scores_by_id,
        corrected_scores_by_id=corrected_scores,
        refusal_score_overrides=forced_scores,
        failed_eval_ids=failed_eval_ids,
        failure_reasons=failure_reasons,
        evaluation_reasons=evaluation_reasons,
        expected_refusal_by_qid=expected_refusal_by_qid,
        refusals_by_qid=refusals_by_qid,
        binary_threshold=BINARY_THRESHOLD,
    )

    return {
        "deepeval_result": result,
        "deepeval_row_ids": row_ids,
        "deepeval_row_count": len(row_ids),
    }


def _compute_deepeval_metrics(
    *,
    deepeval_result: dict[str, object],
    row_ids: list[str],
    total_rows: int,
    expected_refusal_by_qid: dict[str, bool] | None,
    refusals_by_qid: dict[str, bool] | None,
) -> dict[str, object]:
    raw_scores_raw = deepeval_result.get("scores")
    raw_scores_by_id = raw_scores_raw if isinstance(raw_scores_raw, dict) else {}
    corrected_scores_raw = deepeval_result.get("corrected_scores")
    corrected_scores_by_id = corrected_scores_raw if isinstance(corrected_scores_raw, dict) else {}

    deepeval_scores = list(raw_scores_by_id.values())
    deepeval_score_mean = (
        statistics.fmean(float(score) for score in deepeval_scores)
        if deepeval_scores
        else None
    )
    deepeval_binary_labels = [
        1.0 if float(score) >= BINARY_THRESHOLD else 0.0 for score in deepeval_scores
    ]
    deepeval_binary_accuracy = (
        statistics.fmean(deepeval_binary_labels)
        if deepeval_binary_labels
        else None
    )
    deepeval_corrected_scores = list(corrected_scores_by_id.values())
    deepeval_score_mean_corrected = (
        statistics.fmean(float(score) for score in deepeval_corrected_scores)
        if deepeval_corrected_scores
        else None
    )
    deepeval_binary_labels_corrected = [
        1.0 if float(score) >= BINARY_THRESHOLD else 0.0 for score in deepeval_corrected_scores
    ]
    deepeval_binary_accuracy_corrected = (
        statistics.fmean(deepeval_binary_labels_corrected)
        if deepeval_binary_labels_corrected
        else None
    )
    deepeval_coverage = len(deepeval_corrected_scores) / total_rows if total_rows else 0.0
    deepeval_outcome = _compute_outcome_breakdown(
        row_ids=row_ids,
        scores_by_id=raw_scores_by_id,
        expected_refusal_by_qid=expected_refusal_by_qid,
        refusals_by_qid=refusals_by_qid,
        binary_threshold=BINARY_THRESHOLD,
    )
    deepeval_corrected_outcome = _compute_outcome_breakdown(
        row_ids=row_ids,
        scores_by_id=corrected_scores_by_id,
        expected_refusal_by_qid=expected_refusal_by_qid,
        refusals_by_qid=refusals_by_qid,
        binary_threshold=BINARY_THRESHOLD,
    )
    return {
        "deepeval_score_mean": deepeval_score_mean,
        "deepeval_binary_accuracy": deepeval_binary_accuracy,
        "deepeval_score_mean_corrected": deepeval_score_mean_corrected,
        "deepeval_binary_accuracy_corrected": deepeval_binary_accuracy_corrected,
        "deepeval_coverage": deepeval_coverage,
        "deepeval_outcome": deepeval_outcome,
        "deepeval_corrected_outcome": deepeval_corrected_outcome,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate generation outputs with DeepEval correctness."
    )
    parser.add_argument(
        "--generation-dir",
        required=True,
        help="Directory containing generation output JSON files.",
    )
    parser.add_argument(
        "--context-dir",
        default="data/context",
        help="Directory containing context JSON files (default: data/context).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/results",
        help="Base output directory (default: outputs/results).",
    )
    parser.add_argument(
        "--model",
        required=True,
        nargs="+",
        help="One or more model names.",
    )
    parser.add_argument(
        "--setup",
        required=True,
        nargs="+",
        help="One or more setup labels.",
    )
    parser.add_argument(
        "--lang",
        required=True,
        choices=("nl", "fr"),
        help="Dataset language.",
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=f"Dataset id (default: {DEFAULT_DATASET_ID}).",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Dataset split (default: {DEFAULT_SPLIT}).",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=(
            "Judge model config name (from src/generation/model_configs) "
            f"(default: {DEFAULT_JUDGE_MODEL})."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable correctness cache reuse and force recomputation.",
    )
    return parser


def _load_existing_output_payload(*, output_file: Path) -> dict[str, object] | None:
    if not output_file.is_file():
        return None
    try:
        with output_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_null_core_metrics(evaluation: dict[str, Any]) -> bool:
    core_metric_keys = (
        "deepeval_score_mean",
        "deepeval_binary_accuracy",
        "deepeval_score_mean_corrected",
        "deepeval_binary_accuracy_corrected",
        "deepeval_coverage",
    )
    return any(evaluation.get(key) is None for key in core_metric_keys)


def _nan_correctness_row(model_name: str, setup_name: str) -> dict[str, object]:
    return {
        "Model": model_name,
        "Setup": setup_name,
        "ScoreMean": float("nan"),
        "ScoreMeanCorrected": float("nan"),
        "BinAcc": float("nan"),
        "BinAccCorrected": float("nan"),
        "Cov": float("nan"),
        "CorrectAnswers": float("nan"),
        "NonCorrect": float("nan"),
        "CorrectRefusals": float("nan"),
        "IncorrectRefusals": float("nan"),
    }


def main() -> None:
    args = _build_arg_parser().parse_args()
    generation_dir = Path(args.generation_dir)
    context_dir = Path(args.context_dir)
    output_root = Path(args.output_dir)
    if not generation_dir.is_dir():
        raise NotADirectoryError(f"Generation dir not found: {generation_dir}")
    if not context_dir.is_dir():
        raise NotADirectoryError(f"Context dir not found: {context_dir}")

    model_names = [str(model).strip() for model in args.model if str(model).strip()]
    setup_names = [str(setup).strip() for setup in args.setup if str(setup).strip()]
    lang = str(args.lang).strip()
    if not model_names:
        raise ValueError("Resolved model list is empty. Pass --model.")
    if not setup_names:
        raise ValueError("Resolved setup list is empty. Pass --setup.")

    use_cache = not bool(args.no_cache)
    judge_model_arg = str(args.judge_model or DEFAULT_JUDGE_MODEL).strip()
    from generation.cli.judge_selector import load_candidates_from_model_configs

    candidate = load_candidates_from_model_configs(model_names=[judge_model_arg])[0]
    judge_config = candidate["judge_config"]
    judge_model = str(judge_config.get("model") or "").strip() or judge_model_arg
    print(
        f"Resolved judge-model={judge_model_arg} from model configs "
        f"to model={judge_model}."
    )
    print(f"Using judge config: model={judge_model}")

    print(f"Loading gold rows from dataset={args.dataset_id} split={args.split} lang={lang}")
    query_by_qid, ref_by_qid = _load_gold_query_ref_by_qid(
        dataset_id=str(args.dataset_id),
        split=str(args.split),
        lang=lang,
    )
    gold_ids_by_qid = _load_gold_ids_by_qid(
        dataset_id=str(args.dataset_id),
        split=str(args.split),
        lang=lang,
    )
    print(f"Loaded {len(query_by_qid)} gold rows.")

    correctness_rows: list[dict[str, object]] = []
    num_completed_runs = 0

    for setup_name in setup_names:
        context_file = context_dir / lang / f"{setup_name}.json"
        if not context_file.is_file():
            raise FileNotFoundError(f"Context file not found for setup={setup_name}: {context_file}")
        print(f"Loading retrieved context ids from {context_file}")
        retrieved_by_qid = _load_context_data_by_qid(context_file)
        expected_refusal_by_qid = _derive_expected_refusal_by_qid(
            gold_by_qid=gold_ids_by_qid,
            retrieved_by_qid=retrieved_by_qid,
        )

        for model_name in model_names:
            generation_file = generation_dir / lang / setup_name / f"{model_name}.json"
            output_dir = _default_output_dir(
                output_root=output_root,
                lang=lang,
                setup_name=setup_name,
                model_name=model_name,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            deepeval_output_file = output_dir / "deepeval.json"

            if not generation_file.is_file():
                print(
                    f"Missing generation file for setup={setup_name} model={model_name}: "
                    f"returning NaNs (tried: {generation_file})."
                )
                correctness_rows.append(_nan_correctness_row(model_name, setup_name))
                continue

            cached_payload = (
                _load_existing_output_payload(output_file=deepeval_output_file)
                if use_cache
                else None
            )
            if cached_payload is not None:
                print(
                    "Using cached correctness payload: "
                    f"model={model_name} "
                    f"setup={setup_name} "
                    f"file={deepeval_output_file}"
                )

            answers = _load_generation_answers(generation_file)
            prediction_text_by_qid = _load_prediction_text_by_qid(answers)
            refusals_by_qid = _load_refusal_flags_by_qid(answers)
            print(f"Loaded {len(prediction_text_by_qid)} predicted rows from {generation_file}")

            deepeval_eval = run_correctness_evaluation(
                query_by_qid=query_by_qid,
                ref_by_qid=ref_by_qid,
                prediction_text_by_qid=prediction_text_by_qid,
                judge_model=judge_model,
                cached_payload=cached_payload,
                expected_refusal_by_qid=expected_refusal_by_qid,
                refusals_by_qid=refusals_by_qid,
            )
            deepeval_row_ids_raw = deepeval_eval.get("deepeval_row_ids")
            deepeval_row_ids = (
                [str(case_id) for case_id in deepeval_row_ids_raw]
                if isinstance(deepeval_row_ids_raw, list)
                else []
            )
            deepeval_row_count_raw = deepeval_eval.get("deepeval_row_count")
            deepeval_row_count = (
                int(deepeval_row_count_raw)
                if isinstance(deepeval_row_count_raw, (int, float))
                else len(deepeval_row_ids)
            )
            deepeval_result_raw = deepeval_eval.get("deepeval_result")
            deepeval_result = deepeval_result_raw if isinstance(deepeval_result_raw, dict) else {}
            deepeval_metrics = _compute_deepeval_metrics(
                deepeval_result=deepeval_result,
                row_ids=deepeval_row_ids,
                total_rows=deepeval_row_count,
                expected_refusal_by_qid=expected_refusal_by_qid,
                refusals_by_qid=refusals_by_qid,
            )
            if cached_payload is not None and _has_null_core_metrics(deepeval_metrics):
                print(
                    "Cached metrics contained null values after evaluation; "
                    "retrying without cache."
                )
                deepeval_eval = run_correctness_evaluation(
                    query_by_qid=query_by_qid,
                    ref_by_qid=ref_by_qid,
                    prediction_text_by_qid=prediction_text_by_qid,
                    judge_model=judge_model,
                    cached_payload=None,
                    expected_refusal_by_qid=expected_refusal_by_qid,
                    refusals_by_qid=refusals_by_qid,
                )
                deepeval_result_raw = deepeval_eval.get("deepeval_result")
                deepeval_result = deepeval_result_raw if isinstance(deepeval_result_raw, dict) else {}
                deepeval_metrics = _compute_deepeval_metrics(
                    deepeval_result=deepeval_result,
                    row_ids=deepeval_row_ids,
                    total_rows=deepeval_row_count,
                    expected_refusal_by_qid=expected_refusal_by_qid,
                    refusals_by_qid=refusals_by_qid,
                )

            deepeval_score_mean = _as_optional_float(deepeval_metrics.get("deepeval_score_mean"))
            deepeval_binary_accuracy = _as_optional_float(
                deepeval_metrics.get("deepeval_binary_accuracy")
            )
            deepeval_score_mean_corrected = _as_optional_float(
                deepeval_metrics.get("deepeval_score_mean_corrected")
            )
            deepeval_binary_accuracy_corrected = _as_optional_float(
                deepeval_metrics.get("deepeval_binary_accuracy_corrected")
            )
            deepeval_coverage = _as_optional_float(deepeval_metrics.get("deepeval_coverage"))
            deepeval_outcome_raw = deepeval_metrics.get("deepeval_outcome")
            deepeval_outcome = deepeval_outcome_raw if isinstance(deepeval_outcome_raw, dict) else None
            deepeval_corrected_outcome_raw = deepeval_metrics.get("deepeval_corrected_outcome")
            deepeval_corrected_outcome = (
                deepeval_corrected_outcome_raw
                if isinstance(deepeval_corrected_outcome_raw, dict)
                else None
            )
            print(
                "DeepEval metrics: "
                f"model={model_name} "
                f"setup={setup_name} "
                f"coverage={deepeval_coverage} "
                f"score_mean={deepeval_score_mean} "
                f"score_mean_corrected={deepeval_score_mean_corrected} "
                f"binary_accuracy={deepeval_binary_accuracy} "
                f"binary_accuracy_corrected={deepeval_binary_accuracy_corrected}"
            )
            if deepeval_outcome:
                print(
                    "DeepEval outcome: "
                    f"{deepeval_outcome.get('fractions')}"
                )
            if deepeval_corrected_outcome:
                print(
                    "DeepEval corrected outcome: "
                    f"{deepeval_corrected_outcome.get('fractions')}"
                )

            payload = {
                "model": model_name,
                "setup": setup_name,
                "dataset_id": str(args.dataset_id),
                "split": str(args.split),
                "lang": lang,
                "generation_file": str(generation_file),
                "metrics": {
                    "deepeval_score_mean": deepeval_score_mean,
                    "deepeval_binary_accuracy": deepeval_binary_accuracy,
                    "deepeval_score_mean_corrected": deepeval_score_mean_corrected,
                    "deepeval_binary_accuracy_corrected": deepeval_binary_accuracy_corrected,
                    "deepeval_coverage": deepeval_coverage,
                    "deepeval_outcome": deepeval_outcome,
                    "deepeval_corrected_outcome": deepeval_corrected_outcome,
                },
                "deepeval": deepeval_result,
            }
            with deepeval_output_file.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            print(f"Wrote correctness evaluation to {deepeval_output_file}")

            correctness_rows.append(
                {
                    "Model": model_name,
                    "Setup": setup_name,
                    "ScoreMean": deepeval_score_mean,
                    "ScoreMeanCorrected": deepeval_score_mean_corrected,
                    "BinAcc": deepeval_binary_accuracy,
                    "BinAccCorrected": deepeval_binary_accuracy_corrected,
                    "Cov": deepeval_coverage,
                    "CorrectAnswers": (
                        deepeval_corrected_outcome.get("fractions", {}).get("correct_answers")
                        if deepeval_corrected_outcome
                        else None
                    ),
                    "NonCorrect": (
                        deepeval_corrected_outcome.get("fractions", {}).get("non_correct")
                        if deepeval_corrected_outcome
                        else None
                    ),
                    "CorrectRefusals": (
                        deepeval_corrected_outcome.get("fractions", {}).get("correct_refusals")
                        if deepeval_corrected_outcome
                        else None
                    ),
                    "IncorrectRefusals": (
                        deepeval_corrected_outcome.get("fractions", {}).get("incorrect_refusals")
                        if deepeval_corrected_outcome
                        else None
                    ),
                }
            )
            num_completed_runs += 1

    if num_completed_runs == 0:
        print("No real runs were evaluated; all requested rows are NaN.")

    print("All correctness rows:")
    _print_rows(correctness_rows, CORRECTNESS_TABLE_COLUMNS)


if __name__ == "__main__":
    main()
