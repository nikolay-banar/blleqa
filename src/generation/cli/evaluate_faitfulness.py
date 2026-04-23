from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

from datasets import load_dataset
from generation.cli.evaluate_citations import _load_prediction_text_by_qid
from generation.cli.evaluate_refusals import (
    DEFAULT_DATASET_ID,
    DEFAULT_SPLIT,
    _derive_expected_refusal_by_qid,
    _default_output_dir,
    _load_context_data_by_qid,
    _load_generation_answers,
    _load_predicted_ids_by_qid,
    _load_refusal_flags_by_qid,
    _print_rows,
    _refusal_case_label,
)
from generation.evaluators import ragas_faithfulness
from generation.pipeline.blleqa import _load_gold_ids_by_qid, _load_gold_query_ref_by_qid

DEFAULT_JUDGE_MODEL = "gemini-3-flash"
MAX_CONTEXTS_PER_QID = 100
RAGAS_TABLE_COLUMNS = [
    "Model",
    "Setup",
    "RagasFaith",
    "RagasFaithCorrected",
    "RagasCov",
]

def _build_ragas_rows(
    *,
    query_by_qid: dict[str, str],
    prediction_text_by_qid: dict[str, str],
    context_texts_by_qid: dict[str, list[str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    candidate_ids = sorted(
        set(query_by_qid.keys()) & set(prediction_text_by_qid.keys()) & set(context_texts_by_qid.keys())
    )
    for qid in candidate_ids:
        query = query_by_qid.get(qid, "").strip()
        prediction = prediction_text_by_qid.get(qid, "").strip()
        contexts = context_texts_by_qid.get(qid, [])
        if not query or not prediction:
            continue
        rows.append(
            {
                "id": qid,
                "query": query,
                "prediction": prediction,
                "contexts": contexts,
            }
        )
    return rows


def _build_faitfulness_by_id(
    *,
    row_ids: list[str],
    scores_by_id: dict[str, float],
    corrected_scores_by_id: dict[str, float],
    refusal_score_overrides: dict[str, float],
    failed_eval_ids: list[str],
    failure_reasons: dict[str, str],
    expected_refusal_by_qid: dict[str, bool] | None,
    refusals_by_qid: dict[str, bool] | None,
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
            "score": scores_by_id.get(case_id),
            "corrected_score": corrected_scores_by_id.get(case_id),
            "refusal_score_override": refusal_score_overrides.get(case_id),
            "expected_refusal": expected_refusal,
            "predicted_refusal": predicted_refusal,
            "refusal_case": _refusal_case_label(
                expected_refusal=expected_refusal,
                predicted_refusal=predicted_refusal,
            ),
            "failed_eval": case_id in failed_set,
            "failure_reason": failure_reasons.get(case_id),
        }
    return by_id


def run_faitfulness_evaluation(
    *,
    query_by_qid: dict[str, str],
    prediction_text_by_qid: dict[str, str],
    context_texts_by_qid: dict[str, list[str]],
    judge_model: str,
    ragas_batch_size: int,
    cached_payload: dict[str, object] | None,
    expected_refusal_by_qid: dict[str, bool] | None = None,
    refusals_by_qid: dict[str, bool] | None = None,
) -> dict[str, Any]:
    from generation.judge_selection.eval import (
        _evaluate_candidate_correctness,
        _merge_candidate_runs,
        _normalize_candidate_correctness_run,
    )

    ragas_rows = _build_ragas_rows(
        query_by_qid=query_by_qid,
        prediction_text_by_qid=prediction_text_by_qid,
        context_texts_by_qid=context_texts_by_qid,
    )

    candidate = {
        "name": judge_model,
        "judge_config": {"model": judge_model, "batch_size": ragas_batch_size},
    }
    cached_raw: object = {}
    if isinstance(cached_payload, dict):
        cached_raw = cached_payload.get("ragas", {})

    cached_run = _normalize_candidate_correctness_run(
        candidate,
        ragas_rows,
        cached_raw if isinstance(cached_raw, dict) else {},
        fill_missing_failures=False,
    )
    processed_ids = set(cached_run["scores"].keys())
    rows_to_evaluate = [
        row for row in ragas_rows if str(row["id"]) not in processed_ids
    ]
    if processed_ids:
        print(
            "RAGAS cache: "
            f"reusing {len(processed_ids)}/{len(ragas_rows)} scored rows; "
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
            ragas_faithfulness,
        )
    merged_run = _merge_candidate_runs(
        candidate,
        ragas_rows,
        cached_run,
        new_run,
    )

    forced_scores: dict[str, float] = {}
    if expected_refusal_by_qid is not None and refusals_by_qid is not None:
        for row in ragas_rows:
            case_id = str(row["id"])
            expected_refusal = bool(expected_refusal_by_qid.get(case_id, False))
            predicted_refusal = bool(refusals_by_qid.get(case_id, False))
            if predicted_refusal:
                if expected_refusal:
                    forced_scores[case_id] = 1.0
                else:
                    forced_scores[case_id] = 0.0
    if forced_scores:
        print(
            "Faithfulness refusal override: "
            f"forced_scores={len(forced_scores)} "
            "(correct refusal -> 1.0, incorrect refusal -> 0.0)."
        )

    scores_by_id = dict(merged_run["scores"])
    corrected_scores_by_id = dict(scores_by_id)
    corrected_scores_by_id.update(forced_scores)
    failed_eval_ids = [
        case_id for case_id in merged_run["failed_eval_ids"] if case_id not in forced_scores
    ]
    failure_reasons = {
        case_id: reason
        for case_id, reason in merged_run["failure_reasons"].items()
        if case_id not in forced_scores
    }
    result = {
        "scores": scores_by_id,
        "corrected_scores": corrected_scores_by_id,
        "refusal_score_overrides": forced_scores,
        "failed_eval_ids": failed_eval_ids,
        "failure_reasons": failure_reasons,
    }
    row_ids = [str(row["id"]) for row in ragas_rows]
    result["by_id"] = _build_faitfulness_by_id(
        row_ids=row_ids,
        scores_by_id=scores_by_id,
        corrected_scores_by_id=corrected_scores_by_id,
        refusal_score_overrides=forced_scores,
        failed_eval_ids=failed_eval_ids,
        failure_reasons=failure_reasons,
        expected_refusal_by_qid=expected_refusal_by_qid,
        refusals_by_qid=refusals_by_qid,
    )
    return {
        "ragas_result": result,
        "ragas_row_ids": row_ids,
        "ragas_row_count": len(row_ids),
    }


def _compute_ragas_metrics(
    *,
    ragas_result: dict[str, object],
    total_rows: int,
) -> dict[str, object]:
    ragas_scores_raw = ragas_result.get("scores")
    ragas_scores_by_id = ragas_scores_raw if isinstance(ragas_scores_raw, dict) else {}
    ragas_scores = list(ragas_scores_by_id.values())
    ragas_corrected_scores_raw = ragas_result.get("corrected_scores")
    ragas_corrected_scores = (
        list(ragas_corrected_scores_raw.values())
        if isinstance(ragas_corrected_scores_raw, dict)
        else []
    )
    faithfulness = (
        statistics.fmean(float(score) for score in ragas_scores)
        if ragas_scores
        else None
    )
    faithfulness_corrected = (
        statistics.fmean(float(score) for score in ragas_corrected_scores)
        if ragas_corrected_scores
        else None
    )
    coverage = len(ragas_scores) / total_rows if total_rows else 0.0
    return {
        "ragas_faithfulness": faithfulness,
        "ragas_faithfulness_corrected": faithfulness_corrected,
        "ragas_coverage": coverage,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate generation outputs with RAGAS faithfulness."
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
        "--ragas-batch-size",
        type=int,
        default=5,
        help="Batch size used by RAGAS faithfulness evaluation.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable faithfulness cache reuse and force recomputation.",
    )
    return parser


def _load_corpus_text_by_id(*, dataset_id: str, lang: str) -> dict[str, str]:
    corpus = load_dataset(dataset_id, "corpus")[lang]
    corpus_text_by_id: dict[str, str] = {}
    for row in corpus:
        if not isinstance(row, dict):
            continue
        article_id = str(row.get("id") or "").strip()
        if not article_id:
            continue
        reference = str(row.get("reference") or "").strip()
        article = str(row.get("article") or "").strip()
        text = f"{reference}\n{article}"
        if text.strip():
            corpus_text_by_id[article_id] = text
    return corpus_text_by_id


def _build_cited_context_texts_by_qid(
    *,
    predicted_by_qid: dict[str, list[str]],
    corpus_text_by_id: dict[str, str],
) -> dict[str, list[str]]:
    cited_context_texts_by_qid: dict[str, list[str]] = {}
    for qid, cited_ids in predicted_by_qid.items():
        cited_texts: list[str] = []
        seen_texts: dict[str, None] = {}
        for article_id in cited_ids:
            if len(cited_texts) >= MAX_CONTEXTS_PER_QID:
                break
            text = corpus_text_by_id.get(article_id)
            if not text or text in seen_texts:
                continue
            seen_texts[text] = None
            cited_texts.append(text)
        cited_context_texts_by_qid[qid] = cited_texts
    return cited_context_texts_by_qid


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


def _has_null_core_metrics(metrics: dict[str, Any]) -> bool:
    core_metric_keys = (
        "ragas_faithfulness",
        "ragas_faithfulness_corrected",
        "ragas_coverage",
    )
    return any(metrics.get(key) is None for key in core_metric_keys)


def _nan_ragas_row(model_name: str, setup_name: str) -> dict[str, object]:
    return {
        "Model": model_name,
        "Setup": setup_name,
        "RagasFaith": float("nan"),
        "RagasFaithCorrected": float("nan"),
        "RagasCov": float("nan"),
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
    ragas_batch_size = int(args.ragas_batch_size)
    judge_model_arg = str(args.judge_model or DEFAULT_JUDGE_MODEL).strip()
    from generation.cli.judge_selector import load_candidates_from_model_configs

    candidate = load_candidates_from_model_configs(model_names=[judge_model_arg])[0]
    judge_config = candidate["judge_config"]
    judge_model = str(judge_config.get("model") or "").strip() or judge_model_arg
    print(
        f"Resolved judge-model={judge_model_arg} from model configs "
        f"to model={judge_model}."
    )
    print(f"Using judge config: model={judge_model}, batch_size={ragas_batch_size}")

    print(f"Loading gold rows from dataset={args.dataset_id} split={args.split} lang={lang}")
    query_by_qid, _ = _load_gold_query_ref_by_qid(
        dataset_id=str(args.dataset_id),
        split=str(args.split),
        lang=lang,
    )
    gold_ids_by_qid = _load_gold_ids_by_qid(
        dataset_id=str(args.dataset_id),
        split=str(args.split),
        lang=lang,
    )
    print(f"Loading corpus texts from dataset={args.dataset_id} config=corpus lang={lang}")
    corpus_text_by_id = _load_corpus_text_by_id(
        dataset_id=str(args.dataset_id),
        lang=lang,
    )
    print(f"Loaded {len(corpus_text_by_id)} corpus articles.")
    print(f"Loaded {len(query_by_qid)} gold rows.")

    ragas_rows: list[dict[str, object]] = []
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
            ragas_output_file = output_dir / "ragas.json"

            if not generation_file.is_file():
                print(
                    f"Missing generation file for setup={setup_name} model={model_name}: "
                    f"returning NaNs (tried: {generation_file})."
                )
                ragas_rows.append(_nan_ragas_row(model_name, setup_name))
                continue

            cached_payload = (
                _load_existing_output_payload(output_file=ragas_output_file)
                if use_cache
                else None
            )
            if cached_payload is not None:
                print(
                    "Using cached faithfulness payload: "
                    f"model={model_name} "
                    f"setup={setup_name} "
                    f"file={ragas_output_file}"
                )

            answers = _load_generation_answers(generation_file)
            prediction_text_by_qid = _load_prediction_text_by_qid(answers)
            predicted_by_qid = _load_predicted_ids_by_qid(answers)
            context_texts_by_qid = _build_cited_context_texts_by_qid(
                predicted_by_qid=predicted_by_qid,
                corpus_text_by_id=corpus_text_by_id,
            )
            refusals_by_qid = _load_refusal_flags_by_qid(answers)
            print(f"Loaded {len(prediction_text_by_qid)} predicted rows from {generation_file}")

            ragas_eval = run_faitfulness_evaluation(
                query_by_qid=query_by_qid,
                prediction_text_by_qid=prediction_text_by_qid,
                context_texts_by_qid=context_texts_by_qid,
                judge_model=judge_model,
                ragas_batch_size=ragas_batch_size,
                cached_payload=cached_payload,
                expected_refusal_by_qid=expected_refusal_by_qid,
                refusals_by_qid=refusals_by_qid,
            )
            ragas_result_raw = ragas_eval.get("ragas_result")
            ragas_result = ragas_result_raw if isinstance(ragas_result_raw, dict) else {}
            ragas_row_count_raw = ragas_eval.get("ragas_row_count")
            ragas_row_count = (
                int(ragas_row_count_raw)
                if isinstance(ragas_row_count_raw, (int, float))
                else 0
            )
            ragas_metrics = _compute_ragas_metrics(
                ragas_result=ragas_result,
                total_rows=ragas_row_count,
            )
            if cached_payload is not None and _has_null_core_metrics(ragas_metrics):
                print(
                    "Cached metrics contained null values after evaluation; "
                    "retrying without cache."
                )
                ragas_eval = run_faitfulness_evaluation(
                    query_by_qid=query_by_qid,
                    prediction_text_by_qid=prediction_text_by_qid,
                    context_texts_by_qid=context_texts_by_qid,
                    judge_model=judge_model,
                    ragas_batch_size=ragas_batch_size,
                    cached_payload=None,
                    expected_refusal_by_qid=expected_refusal_by_qid,
                    refusals_by_qid=refusals_by_qid,
                )
                ragas_result_raw = ragas_eval.get("ragas_result")
                ragas_result = ragas_result_raw if isinstance(ragas_result_raw, dict) else {}
                ragas_metrics = _compute_ragas_metrics(
                    ragas_result=ragas_result,
                    total_rows=ragas_row_count,
                )

            ragas_faithfulness = _as_optional_float(ragas_metrics.get("ragas_faithfulness"))
            ragas_faithfulness_corrected = _as_optional_float(
                ragas_metrics.get("ragas_faithfulness_corrected")
            )
            ragas_coverage = _as_optional_float(ragas_metrics.get("ragas_coverage"))
            print(
                "RAGAS metrics: "
                f"model={model_name} "
                f"setup={setup_name} "
                f"faithfulness={ragas_faithfulness} "
                f"faithfulness_corrected={ragas_faithfulness_corrected} "
                f"coverage={ragas_coverage}"
            )

            payload = {
                "model": model_name,
                "setup": setup_name,
                "dataset_id": str(args.dataset_id),
                "split": str(args.split),
                "lang": lang,
                "generation_file": str(generation_file),
                "context_file": str(context_file),
                "metrics": {
                    "ragas_faithfulness": ragas_faithfulness,
                    "ragas_faithfulness_corrected": ragas_faithfulness_corrected,
                    "ragas_coverage": ragas_coverage,
                },
                "ragas": ragas_result,
            }
            with ragas_output_file.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            print(f"Wrote faithfulness evaluation to {ragas_output_file}")

            ragas_rows.append(
                {
                    "Model": model_name,
                    "Setup": setup_name,
                    "RagasFaith": ragas_faithfulness,
                    "RagasFaithCorrected": ragas_faithfulness_corrected,
                    "RagasCov": ragas_coverage,
                }
            )
            num_completed_runs += 1

    if num_completed_runs == 0:
        print("No real runs were evaluated; all requested rows are NaN.")

    print("All faithfulness rows:")
    _print_rows(ragas_rows, RAGAS_TABLE_COLUMNS)

    import pandas as pd
    pd.DataFrame(ragas_rows).to_csv(f"fait_{lang}.csv", index=False)



if __name__ == "__main__":
    main()
