import argparse
import json
import statistics
import warnings
from pathlib import Path
from typing import Callable, TypedDict

import pandas as pd

from generation.evaluators.deepeval_metrics import (
    CorrectnessInput,
    CorrectnessResult,
    JudgeConfig,
    deepeval_correctness,
)
from generation.evaluators.judge_metrics import (
    f1_macro_score,
    mean_absolute_error,
    pearson_correlation,
    spearman_correlation,
)

MODEL_CONFIG_DIR = Path(__file__).resolve().parent / "model_configs"
DEFAULT_ID_COL = "id"
DEFAULT_QUERY_COL = "questions"
DEFAULT_PREDICTION_COL = "llm_answers"
DEFAULT_REF_COL = "gold_answers"
DEFAULT_GRADE_COL = "grade"


class JudgeCandidate(TypedDict):
    name: str
    judge_config: JudgeConfig


class JudgeEvaluation(TypedDict):
    name: str
    judge_config: JudgeConfig
    coverage: float
    mean_score: float | None
    score_std: float | None
    failure_rate: float
    num_scored: int
    num_failed: int
    num_compared_with_gold: int
    pearson_correlation: float | None
    spearman_correlation: float | None
    mae: float | None
    f1_macro: float | None
    failed_eval_ids: list[str]
    failure_reasons: dict[str, str]


class JudgeSelectionResult(TypedDict):
    selected_judge_name: str
    selected_judge_config: JudgeConfig
    ranking: list[JudgeEvaluation]


class CandidateCorrectnessRun(TypedDict):
    name: str
    judge_config: JudgeConfig
    scores: dict[str, float]
    failed_eval_ids: list[str]
    failure_reasons: dict[str, str]
    evaluation_reasons: dict[str, str]


EvaluatorFn = Callable[[list[CorrectnessInput], JudgeConfig | None], CorrectnessResult]

def load_candidates_from_model_configs(
    *,
    model_names: list[str] | None = None,
    config_dir: Path = MODEL_CONFIG_DIR,
) -> list[JudgeCandidate]:
    config_paths = sorted(config_dir.glob("*.json"))
    if not config_paths:
        raise ValueError(f"No model config files found in {config_dir}")

    requested_names: set[str] | None = None
    if model_names is not None:
        requested_names = {
            name.strip()[:-5] if name.strip().endswith(".json") else name.strip()
            for name in model_names
            if name.strip()
        }
        if not requested_names:
            raise ValueError("model_names must contain at least one non-empty model name")
    else:
        requested_names = {path.stem for path in config_paths}

    candidates: list[JudgeCandidate] = []
    for model_name in sorted(requested_names):
        config_path = config_dir / f"{model_name}.json"
        if not config_path.is_file():
            raise ValueError(
                f"No model config found for requested candidate: {model_name} in {config_dir}"
            )

        with config_path.open("r", encoding="utf-8") as file:
            config_data = json.load(file)

        configured_model = str((config_data.get("name") or config_data.get("model") or "")).strip()
        if not configured_model:
            raise ValueError(f"Missing 'name' (or 'model') in model config: {config_path}")

        judge_config: JudgeConfig = {"model": configured_model}
        providers = config_data.get("providers")
        if isinstance(providers, list):
            provider_list = [str(provider).strip() for provider in providers if str(provider).strip()]
            if provider_list:
                judge_config["providers"] = provider_list

        base_url = config_data.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            judge_config["base_url"] = base_url.strip()

        api_key = config_data.get("api_key")
        if isinstance(api_key, str) and api_key.strip():
            judge_config["api_key"] = api_key.strip()

        candidates.append({"name": model_name, "judge_config": judge_config})


    return candidates


def _load_csv_records(
    csv_path: str | Path,
    *,
    id_col: str,
    query_col: str,
    prediction_col: str,
    ref_col: str,
    gold_score_col: str | None = None,
) -> list[dict[str, object]]:
    path = Path(csv_path)

    if not path.exists():
        raise ValueError(f"CSV path not found: {path}")
    if path.is_dir():
        raise ValueError(f"Expected a single CSV file path, got directory: {path}")

    dataframe = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    records = dataframe.to_dict(orient="records")

    normalized_rows: list[dict[str, object]] = []
    for index, row in enumerate(records, start=1):
        case_id = str((row.get(id_col) or "")).strip()
        if not case_id:
            raise ValueError(
                f"Row {index} in {path} has empty required field ({id_col})."
            )
        query = str((row.get(query_col) or "")).strip()
        prediction = str((row.get(prediction_col) or "")).strip()
        reference = str((row.get(ref_col) or "")).strip()
        if not query or not reference:
            raise ValueError(
                f"Row {index} in {path} has empty required fields "
                f"({query_col}, {ref_col})."
            )
        gold_score: float | None = None
        if gold_score_col is not None:
            raw_gold_score = str((row.get(gold_score_col) or "")).strip()
            if raw_gold_score:
                try:
                    gold_score = float(raw_gold_score)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid gold score at row {index} in {path}: {raw_gold_score}"
                    ) from exc
        if not prediction:
            warnings.warn(
                f"Row {index} in {path} is missing '{prediction_col}' and will be skipped.",
                stacklevel=2,
            )
            continue
        normalized_row: dict[str, object] = {
            "id": case_id,
            "query": query,
            "prediction": prediction,
            "ref": reference,
        }
        if gold_score_col is not None:
            normalized_row["gold_score"] = gold_score
        normalized_rows.append(normalized_row)
    return normalized_rows


def load_correctness_rows_from_json(
    output_path: str | Path,
    *,
    model_name: str,
    scores_col: str = "scores",
) -> CorrectnessResult:
    path = Path(output_path)

    if not model_name:
        raise ValueError("model_name must be non-empty")

    candidate_json_path = path / f"{model_name}.json"

    if not candidate_json_path.exists():
        return {
            "scores": {},
            "failed_eval_ids": [],
            "failure_reasons": {},
            "evaluation_reasons": {},
        }

    with candidate_json_path.open("r", encoding="utf-8") as file:
        try:
            payload = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in cached file: {candidate_json_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in cached file: {candidate_json_path}")

    raw_scores = payload.get(scores_col, {})
    if not isinstance(raw_scores, dict):
        raise ValueError(
            f"Expected '{scores_col}' to be an object in cached file: {candidate_json_path}"
        )
    raw_failed_eval_ids = payload.get("failed_eval_ids", [])
    if not isinstance(raw_failed_eval_ids, list):
        raise ValueError(
            f"Expected 'failed_eval_ids' to be a list in cached file: {candidate_json_path}"
        )
    raw_failure_reasons = payload.get("failure_reasons", {})
    if not isinstance(raw_failure_reasons, dict):
        raise ValueError(
            f"Expected 'failure_reasons' to be an object in cached file: {candidate_json_path}"
        )
    raw_evaluation_reasons = payload.get("evaluation_reasons", {})
    if not isinstance(raw_evaluation_reasons, dict):
        raise ValueError(
            f"Expected 'evaluation_reasons' to be an object in cached file: {candidate_json_path}"
        )

    return {
        "scores": raw_scores,
        "failed_eval_ids": raw_failed_eval_ids,
        "failure_reasons": raw_failure_reasons,
        "evaluation_reasons": raw_evaluation_reasons,
    }


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the best DeepEval judge model from candidates.")
    parser.add_argument(
        "--csv-path",
        required=True,
        help="Path used to derive annotation name (folder stem under judge_selection).",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        required=True,
        help="Model names to evaluate; matching config files are loaded from the model config directory.",
    )
    parser.add_argument(
        "--config-dir",
        default=str(MODEL_CONFIG_DIR),
        help="Directory containing model config JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="../outputs/judge_selection",
        help="Output base directory containing judge_selection/{annotation_name}/{model}.jsonl.",
    )
    return parser.parse_args(argv)


def run_cli(args: argparse.Namespace) -> JudgeSelectionResult:
    candidates = load_candidates_from_model_configs(
        model_names=args.candidates,
        config_dir=Path(args.config_dir),
    )


    output_dir = Path(args.output_dir)
    judge_selection_root = output_dir if output_dir.name == "judge_selection" else output_dir / "judge_selection"
    judge_selection_root.mkdir(parents=True, exist_ok=True)
    # print(judge_selection_root)
    # output_path = judge_selection_root / "judge_selection.json"

    input_rows = _load_csv_records(
        args.csv_path,
        id_col=DEFAULT_ID_COL,
        query_col=DEFAULT_QUERY_COL,
        prediction_col=DEFAULT_PREDICTION_COL,
        ref_col=DEFAULT_REF_COL,
        gold_score_col=DEFAULT_GRADE_COL,
    )
    gold_scores_by_id: dict[str, float] = {
        str(row["id"]): float(row["gold_score"])
        for row in input_rows
        if row.get("gold_score") is not None
    }
    expected_id_list = [str(row["id"]) for row in input_rows]
    inputs_by_id = {row["id"]: row for row in input_rows}

    annotation_name = Path(args.csv_path).stem
    per_model_output_dir = judge_selection_root / annotation_name
    per_model_output_dir.mkdir(parents=True, exist_ok=True)


    candidate_runs: list[CandidateCorrectnessRun] = []
    for candidate in candidates:
        cached_result = load_correctness_rows_from_json(
            per_model_output_dir,
            model_name=candidate["name"],
        )
        cached_run = _normalize_candidate_correctness_run(
            candidate,
            input_rows,
            cached_result,
            fill_missing_failures=False,
        )
        processed_ids = set(cached_run["scores"].keys())
        missing_experiments = [
            row
            for row in input_rows
            if row["id"] not in processed_ids
        ]
        # missing_experiments = missing_experiments[:5]
        conducted_new_experiments = bool(missing_experiments)
        print("Experiments left: ", len(missing_experiments))

        if not missing_experiments:
            print(
                f"[{candidate['name']}] already fully cached "
                f"({len(processed_ids)}/{len(expected_id_list)} rows)."
            )
            final_run = cached_run
        else:
            new_run = _evaluate_candidate_correctness(
                candidate=candidate,
                input_rows=missing_experiments,
                evaluator=deepeval_correctness,
            )
              
            if not processed_ids:
                final_run = new_run
            else:
                final_run = _merge_candidate_runs(
                    candidate=candidate,
                    input_rows=input_rows,
                    cached_run=cached_run,
                    new_run=new_run
                )
        candidate_runs.append(final_run)
        has_all_experiments = len(final_run["scores"]) == len(expected_id_list)
        model_metrics = None
        if has_all_experiments:
            model_metrics = _compute_judge_evaluation(
                candidate_run=final_run,
                input_rows=input_rows,
                score_range=(1.0, 5.0),
                gold_scores_by_id=gold_scores_by_id,
            )

        if not conducted_new_experiments:
            continue

        failed_id_set = set(final_run["failed_eval_ids"])
        evaluations = [
            {
                "id": case_id,
                "input": inputs_by_id.get(case_id, {}),
                "correctness": final_run["scores"].get(case_id),
                "failed": case_id in failed_id_set,
                "failure_reason": final_run["failure_reasons"].get(case_id),
                "evaluation_reason": final_run["evaluation_reasons"].get(case_id),
            }
            for case_id in expected_id_list
        ]

        model_output_path = per_model_output_dir / f"{final_run['name']}.json"
        model_payload = {
            "input_path": str(args.csv_path),
            "candidate_name": final_run["name"],
            "judge_config": final_run["judge_config"],
            "metrics": model_metrics,
            "scores": final_run["scores"],
            "failed_eval_ids": final_run["failed_eval_ids"],
            "failure_reasons": final_run["failure_reasons"],
            "evaluation_reasons": final_run["evaluation_reasons"],
            "evaluations": evaluations,
        }
        with model_output_path.open("w", encoding="utf-8") as file:
            json.dump(model_payload, file, indent=2, ensure_ascii=False)

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_cli(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
