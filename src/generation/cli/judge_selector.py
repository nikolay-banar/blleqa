import argparse
import json
import logging
import warnings
from pathlib import Path

import pandas as pd

from generation.evaluators.deepeval_metrics import (
    JudgeConfig,
    deepeval_correctness,
)
from generation.judge_selection.eval import ( 
    _evaluate_candidate_correctness,
    _merge_candidate_runs,
    _compute_judge_evaluation,
    _normalize_candidate_correctness_run)

from generation.judge_selection.schema import JudgeCandidate, CandidateCorrectnessRun

logger = logging.getLogger(__name__)

MODEL_CONFIG_DIR = Path(__file__).resolve().parents[1] / "model_configs"
DEFAULT_ID_COL = "id"
DEFAULT_QUERY_COL = "questions"
DEFAULT_PREDICTION_COL = "llm_answers"
DEFAULT_REF_COL = "gold_answers"
DEFAULT_GRADE_COL = "grade"


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


def _load_annotations(
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
            else:
                warnings.warn(
                    f"Row {index} (case id: {case_id}) in {path} is missing '{gold_score_col}' and will be skipped.",
                    stacklevel=2,
                )
                continue

        if not prediction:
            warnings.warn(
                f"Row {index} (case id: {case_id}) in {path} is missing '{prediction_col}' and will be skipped.",
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
) -> dict[str, object]:
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
    result: dict[str, object] = {
        "scores": raw_scores,
        "failed_eval_ids": raw_failed_eval_ids,
        "failure_reasons": raw_failure_reasons,
        "evaluation_reasons": raw_evaluation_reasons,
    }
    if "metrics" in payload:
        raw_metrics = payload.get("metrics")
        if raw_metrics is not None and not isinstance(raw_metrics, dict):
            raise ValueError(
                f"Expected 'metrics' to be an object or null in cached file: {candidate_json_path}"
            )
        result["metrics"] = raw_metrics

    return result


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
        default="outputs/judge_selection",
        help="Output base directory containing judge_selection/{annotation_name}/{model}.jsonl.",
    )
    return parser.parse_args(argv)


def run_cli(args: argparse.Namespace):
    candidates = load_candidates_from_model_configs(
        model_names=args.candidates,
        config_dir=Path(args.config_dir),
    )


    output_dir = Path(args.output_dir)
    judge_selection_root = output_dir if output_dir.name == "judge_selection" else output_dir / "judge_selection"
    judge_selection_root.mkdir(parents=True, exist_ok=True)
    input_rows = _load_annotations(
        args.csv_path,
        id_col=DEFAULT_ID_COL,
        query_col=DEFAULT_QUERY_COL,
        prediction_col=DEFAULT_PREDICTION_COL,
        ref_col=DEFAULT_REF_COL,
        gold_score_col=DEFAULT_GRADE_COL,
    )
    logger.info("Loaded %s input rows from %s.", len(input_rows), args.csv_path)
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


    metrics_by_judge: dict[str, object] = {}
    for candidate in candidates:
        cached_result = load_correctness_rows_from_json(
            per_model_output_dir,
            model_name=candidate["name"],
        )
        model_metrics = cached_result.get("metrics") if isinstance(cached_result.get("metrics"), dict) else None
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
        logger.info("Experiments left: %s", len(missing_experiments))

        if not missing_experiments:
            logger.info(
                "[%s] already fully cached (%s/%s rows).",
                candidate["name"],
                len(processed_ids),
                len(expected_id_list),
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
        has_all_experiments = len(final_run["scores"]) == len(expected_id_list)
        if has_all_experiments and model_metrics is None:
            model_metrics = _compute_judge_evaluation(
                candidate_run=final_run,
                input_rows=input_rows,
                score_range=(1.0, 5.0),
                gold_scores_by_id=gold_scores_by_id,
            )
            logger.info("[%s] computed metrics: %s", candidate["name"], model_metrics)
        elif has_all_experiments:
            logger.info("[%s] using cached metrics: %s", candidate["name"], model_metrics)

        metrics_by_judge[candidate["name"]] = model_metrics

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

    metrics_rows: list[dict[str, object]] = []
    for judge_name, metrics in sorted(metrics_by_judge.items()):
        row: dict[str, object] = {"judge": judge_name}
        if isinstance(metrics, dict):
            row.update(
                {
                    "coverage": metrics.get("coverage"),
                    "mean_score": metrics.get("mean_score"),
                    "score_std": metrics.get("score_std"),
                    "failure_rate": metrics.get("failure_rate"),
                    "num_scored": metrics.get("num_scored"),
                    "num_failed": metrics.get("num_failed"),
                    "num_compared_with_gold": metrics.get("num_compared_with_gold"),
                    "pearson_correlation": metrics.get("pearson_correlation"),
                    "spearman_correlation": metrics.get("spearman_correlation"),
                    "mae": metrics.get("mae"),
                    "f1_macro": metrics.get("f1_macro"),
                }
            )
        else:
            row["metrics"] = metrics
        metrics_rows.append(row)

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        if "pearson_correlation" in metrics_df.columns:
            metrics_df = metrics_df.sort_values(
                by="pearson_correlation",
                ascending=False,
                na_position="last",
            )
        print(metrics_df.to_string(index=False))
    else:
        print("No metrics available.")

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args(argv)
    run_cli(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
