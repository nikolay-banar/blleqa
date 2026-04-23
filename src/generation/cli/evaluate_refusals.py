from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from generation.evaluators.refusal import _evaluate_refusals, _is_refusal_output
from generation.pipeline.blleqa import _load_gold_ids_by_qid
from generation.pipeline.blleqa import _to_id_list

DEFAULT_DATASET_ID = "clips/bLLeQA"
DEFAULT_SPLIT = "test"
REFUSAL_TABLE_COLUMNS = [
    "Model",
    "Setup",
    "RefRate",
    "F1-macro",
    "RefPr",
    "RefRec",
    "RefF1",
    "NonRefRPr",
    "NonRefRRec",
    "NonRefRF1",
]


def _refusal_case_label(*, expected_refusal: bool | None, predicted_refusal: bool | None) -> str | None:
    if expected_refusal is None or predicted_refusal is None:
        return None
    if expected_refusal and predicted_refusal:
        return "correct_refusal"
    if expected_refusal and not predicted_refusal:
        return "missed_refusal"
    if not expected_refusal and predicted_refusal:
        return "incorrect_refusal"
    return "answer_expected_and_answered"


def _to_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _extract_refusal_eval_from_payload(payload: dict[str, object]) -> dict[str, Any] | None:
    metrics = _to_dict(payload.get("metrics"))
    counts = _to_dict(payload.get("counts"))
    refusal = _to_dict(payload.get("refusal"))
    if not refusal and not metrics and not counts:
        return None
    refusal_result = {
        **refusal,
        "metrics": metrics,
        "counts": counts,
    }
    return {
        "refusal_result": refusal_result,
        "metrics": metrics,
        "counts": counts,
    }


def load_cached_refusal_evaluation(*, refusal_output_file: Path) -> dict[str, Any] | None:
    if not refusal_output_file.is_file():
        return None
    try:
        with refusal_output_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return _extract_refusal_eval_from_payload(payload)


def build_refusal_payload(
    *,
    model_name: str,
    setup_name: str,
    dataset_id: str,
    split: str,
    lang: str,
    generation_file: Path,
    context_file: Path,
    refusal_eval: dict[str, Any],
) -> dict[str, Any]:
    refusal_result_raw = refusal_eval.get("refusal_result", {})
    refusal_result = refusal_result_raw if isinstance(refusal_result_raw, dict) else {}
    metrics_raw = refusal_eval.get("metrics", {})
    counts_raw = refusal_eval.get("counts", {})
    metrics = metrics_raw if isinstance(metrics_raw, dict) else {}
    counts = counts_raw if isinstance(counts_raw, dict) else {}
    return {
        "model": model_name,
        "setup": setup_name,
        "dataset_id": dataset_id,
        "split": split,
        "lang": lang,
        "generation_file": str(generation_file),
        "context_file": str(context_file),
        "metrics": metrics,
        "counts": counts,
        "refusal": {
            key: value
            for key, value in refusal_result.items()
            if key not in {"metrics", "counts"}
        },
    }


def run_refusal_evaluation(
    *,
    gold_by_qid: dict[str, list[str]],
    expected_refusal_by_qid: dict[str, bool],
    retrieved_by_qid: dict[str, list[str]],
    model_name: str,
    setup_name: str,
    predicted_by_qid: dict[str, list[str]] | None = None,
    refusals_by_qid: dict[str, bool] | None = None,
    generation_file: Path | None = None,
    refusal_output_file: Path | None = None,
    dataset_id: str | None = None,
    split: str | None = None,
    lang: str | None = None,
    context_file: Path | None = None,
    persist_output: bool = False,
) -> dict[str, Any]:
    resolved_predicted_by_qid = predicted_by_qid
    resolved_refusals_by_qid = refusals_by_qid
    if resolved_predicted_by_qid is None or resolved_refusals_by_qid is None:
        if generation_file is None:
            raise ValueError(
                "run_refusal_evaluation requires either predicted/refusal maps "
                "or generation_file."
            )
        answers = _load_generation_answers(generation_file)
        resolved_predicted_by_qid = _load_predicted_ids_by_qid(answers)
        resolved_refusals_by_qid = _load_refusal_flags_by_qid(answers)
        print(f"Loaded {len(resolved_predicted_by_qid)} predicted rows from {generation_file}")

    refusal_result = _evaluate_refusals(
        gold_by_qid=gold_by_qid,
        expected_refusal_by_qid=expected_refusal_by_qid,
        retrieved_by_qid=retrieved_by_qid,
        predicted_by_qid=resolved_predicted_by_qid,
        refusals_by_qid=resolved_refusals_by_qid,
        model_name=model_name,
        setup_name=setup_name,
    )
    metrics = refusal_result.get("metrics", {})
    counts = refusal_result.get("counts", {})
    refusal_eval = {
        "refusal_result": refusal_result,
        "metrics": metrics if isinstance(metrics, dict) else {},
        "counts": counts if isinstance(counts, dict) else {},
    }
    if persist_output:
        if (
            refusal_output_file is None
            or generation_file is None
            or dataset_id is None
            or split is None
            or lang is None
            or context_file is None
        ):
            raise ValueError(
                "persist_output=True requires refusal_output_file, generation_file, "
                "dataset_id, split, lang, and context_file."
            )
        metrics_dict = _to_dict(refusal_eval.get("metrics"))
        print(
            "Refusal metrics: "
            f"model={model_name} "
            f"setup={setup_name} "
            f"accuracy={metrics_dict.get('refusal_accuracy')} "
            f"pred_rate={metrics_dict.get('refusal_rate_predicted')} "
            f"compared_rate={metrics_dict.get('refusal_rate_compared')}"
        )
        payload = build_refusal_payload(
            model_name=model_name,
            setup_name=setup_name,
            dataset_id=dataset_id,
            split=split,
            lang=lang,
            generation_file=generation_file,
            context_file=context_file,
            refusal_eval=refusal_eval,
        )
        with refusal_output_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"Wrote refusal evaluation to {refusal_output_file}")
    return refusal_eval


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate generation outputs with refusal metrics only."
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
        "--no-cache",
        action="store_true",
        help="Disable refusal cache reuse and force recomputation.",
    )
    return parser


def _load_context_data_by_qid(
    context_file: Path,
) -> dict[str, list[str]]:
    with context_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Context file must contain a JSON object keyed by question id.")

    retrieved_by_qid: dict[str, list[str]] = {}
    for qid_raw, value in payload.items():
        qid = str(qid_raw).strip()
        if not qid:
            continue

        ids: list[str] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    article_id = str(item.get("id") or "").strip()
                    if article_id:
                        ids.append(article_id)
                else:
                    article_id = str(item).strip()
                    if article_id:
                        ids.append(article_id)
        elif isinstance(value, dict):
            for article_id_raw in value.keys():
                article_id = str(article_id_raw).strip()
                if article_id:
                    ids.append(article_id)

        unique_ids: dict[str, None] = {}
        for article_id in ids:
            unique_ids[article_id] = None
        retrieved_by_qid[qid] = list(unique_ids.keys())
    return retrieved_by_qid


def _derive_expected_refusal_by_qid(
    *,
    gold_by_qid: dict[str, list[str]],
    retrieved_by_qid: dict[str, list[str]],
) -> dict[str, bool]:
    expected_refusal_by_qid: dict[str, bool] = {}
    for qid, gold_ids in gold_by_qid.items():
        expected_refusal_by_qid[qid] = not set(gold_ids).issubset(set(retrieved_by_qid.get(qid, [])))
    return expected_refusal_by_qid


def _load_generation_answers(generation_file: Path) -> list[dict[str, object]]:
    with generation_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Generation file must contain a JSON object.")

    answers = payload.get("answers", [])
    if not isinstance(answers, list):
        raise ValueError("Generation file must contain 'answers' as a list.")
    return [row for row in answers if isinstance(row, dict)]


def _load_predicted_ids_by_qid(
    answers: list[dict[str, object]],
) -> dict[str, list[str]]:
    predicted_by_qid: dict[str, list[str]] = {}
    for row in answers:
        qid = str(row.get("id") or "").strip()
        if not qid:
            continue

        cleaned = row.get("cleaned", [])
        unique_ids: dict[str, None] = {}
        if isinstance(cleaned, list):
            for chunk in cleaned:
                if not isinstance(chunk, dict):
                    continue
                for source_id in _to_id_list(chunk.get("supported_sources")):
                    unique_ids[source_id] = None
        predicted_by_qid[qid] = list(unique_ids.keys())
    return predicted_by_qid


def _load_refusal_flags_by_qid(
    answers: list[dict[str, object]],
) -> dict[str, bool]:
    refusals_by_qid: dict[str, bool] = {}
    for row in answers:
        qid = str(row.get("id") or "").strip()
        if not qid:
            continue
        refusals_by_qid[qid] = _is_refusal_output(row.get("cleaned", []))
    return refusals_by_qid


def _default_output_dir(*, output_root: Path, lang: str, setup_name: str, model_name: str) -> Path:
    return output_root / lang / setup_name / model_name


def _nan_refusal_row(model_name: str, setup_name: str) -> dict[str, object]:
    return {
        "Model": model_name,
        "Setup": setup_name,
        "RefRate": float("nan"),
        "F1-macro": float("nan"),
        "RefPr": float("nan"),
        "RefRec": float("nan"),
        "RefF1": float("nan"),
        "NonRefRPr": float("nan"),
        "NonRefRRec": float("nan"),
        "NonRefRF1": float("nan"),
    }


def _print_rows(rows: list[dict[str, object]], columns: list[str]) -> None:
    if not rows:
        return
    available_columns = [column for column in columns if any(column in row for row in rows)]
    if not available_columns:
        print("No rows to print.")
        return

    def _format_value(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            # Keep terminal tables readable without changing persisted JSON precision.
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value)

    widths = {
        column: max(
            len(column),
            max(len(_format_value(row.get(column, ""))) for row in rows),
        )
        for column in available_columns
    }
    header = " ".join(column.ljust(widths[column]) for column in available_columns)
    separator = " ".join("-" * widths[column] for column in available_columns)
    print(header)
    print(separator)
    for row in rows:
        line = " ".join(
            _format_value(row.get(column, "")).ljust(widths[column]) for column in available_columns
        )
        print(line)


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

    print(f"Loading gold ids from dataset={args.dataset_id} split={args.split} lang={lang}")
    gold_by_qid = _load_gold_ids_by_qid(
        dataset_id=str(args.dataset_id),
        split=str(args.split),
        lang=lang,
    )
    print(f"Loaded {len(gold_by_qid)} gold rows.")

    refusal_rows: list[dict[str, object]] = []
    num_completed_runs = 0

    for setup_name in setup_names:
        context_file = context_dir / lang / f"{setup_name}.json"
        if not context_file.is_file():
            raise FileNotFoundError(f"Context file not found for setup={setup_name}: {context_file}")
        print(f"Loading retrieved context ids from {context_file}")
        retrieved_by_qid = _load_context_data_by_qid(context_file)
        expected_refusal_by_qid = _derive_expected_refusal_by_qid(
            gold_by_qid=gold_by_qid,
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
            refusal_output_file = output_dir / "refusal.json"

            if not generation_file.is_file():
                print(
                    f"Missing generation file for setup={setup_name} model={model_name}: "
                    f"returning NaNs (tried: {generation_file})."
                )
                refusal_rows.append(_nan_refusal_row(model_name, setup_name))
                continue

            if use_cache:
                cached_refusal_eval = load_cached_refusal_evaluation(
                    refusal_output_file=refusal_output_file
                )
                if cached_refusal_eval is not None:
                    print(
                        "Using cached refusal metrics: "
                        f"model={model_name} "
                        f"setup={setup_name} "
                        f"file={refusal_output_file}"
                    )
                    cached_metrics = _to_dict(cached_refusal_eval.get("metrics"))
                    cached_refusal_table = cached_metrics.get("refusal_table")
                    if isinstance(cached_refusal_table, dict):
                        refusal_rows.append(cached_refusal_table)
                    else:
                        refusal_rows.append(_nan_refusal_row(model_name, setup_name))
                    num_completed_runs += 1
                    continue

            refusal_eval = run_refusal_evaluation(
                generation_file=generation_file,
                gold_by_qid=gold_by_qid,
                expected_refusal_by_qid=expected_refusal_by_qid,
                retrieved_by_qid=retrieved_by_qid,
                model_name=model_name,
                setup_name=setup_name,
                dataset_id=str(args.dataset_id),
                split=str(args.split),
                lang=lang,
                context_file=context_file,
                refusal_output_file=refusal_output_file,
                persist_output=True,
            )

            metrics = refusal_eval.get("metrics", {})
            metrics_dict = metrics if isinstance(metrics, dict) else {}
            refusal_table = metrics_dict.get("refusal_table")
            if isinstance(refusal_table, dict):
                refusal_rows.append(refusal_table)
            else:
                refusal_rows.append(_nan_refusal_row(model_name, setup_name))
            num_completed_runs += 1

    if num_completed_runs == 0:
        print("No real runs were evaluated; all requested rows are NaN.")

    print("All refusal rows:")
    _print_rows(refusal_rows, REFUSAL_TABLE_COLUMNS)


if __name__ == "__main__":
    main()
