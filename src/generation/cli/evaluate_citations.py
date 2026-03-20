from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from generation.cli.evaluate_refusals import (
    DEFAULT_DATASET_ID,
    DEFAULT_SPLIT,
    _default_output_dir,
    _derive_expected_refusal_by_qid,
    _load_context_data_by_qid,
    _load_generation_answers,
    _load_predicted_ids_by_qid,
    _load_refusal_flags_by_qid,
    run_refusal_evaluation,
)
from generation.evaluators.citation_metrics import _evaluate_citations
from generation.pipeline.bbleqa import _load_gold_ids_by_qid

CITATION_TABLE_COLUMNS = [
    "Model",
    "Setup",
    "CitPr",
    "CitRec",
    "CitF1",
]


def _to_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _extract_citation_eval_from_payload(payload: dict[str, object]) -> dict[str, Any] | None:
    metrics = _to_dict(payload.get("metrics"))
    counts = _to_dict(payload.get("counts"))
    citation = _to_dict(payload.get("citation"))
    if not citation and not metrics and not counts:
        return None
    citation_result = {
        **citation,
        "metrics": metrics,
        "counts": counts,
    }
    return {
        "citation_result": citation_result,
        "metrics": metrics,
        "counts": counts,
    }


def load_cached_citation_evaluation(*, citation_output_file: Path) -> dict[str, Any] | None:
    if not citation_output_file.is_file():
        return None
    try:
        with citation_output_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return _extract_citation_eval_from_payload(payload)


def build_citation_payload(
    *,
    model_name: str,
    setup_name: str,
    dataset_id: str,
    split: str,
    lang: str,
    generation_file: Path,
    context_file: Path,
    citation_result: dict[str, object],
    metrics: dict[str, object],
) -> dict[str, object]:
    citation_metrics_payload = citation_result.get("metrics", {})
    citation_table = metrics.get("citation_table")
    if isinstance(citation_metrics_payload, dict) and isinstance(citation_table, dict):
        citation_metrics_payload = {
            **citation_metrics_payload,
            "citation_table": citation_table,
        }
    return {
        "model": model_name,
        "setup": setup_name,
        "dataset_id": dataset_id,
        "split": split,
        "lang": lang,
        "generation_file": str(generation_file),
        "context_file": str(context_file),
        "metrics": citation_metrics_payload,
        "counts": citation_result.get("counts", {}),
        "citation": {
            key: value
            for key, value in citation_result.items()
            if key not in {"metrics", "counts"}
        },
    }


def _cleaned_to_prediction_text(cleaned: object) -> str:
    if isinstance(cleaned, list):
        parts: list[str] = []
        for item in cleaned:
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                text = item.strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(cleaned, dict):
        text = str(cleaned.get("text") or "").strip()
        if text:
            return text
    if isinstance(cleaned, str):
        return cleaned.strip()
    return ""


def _load_prediction_text_by_qid(
    answers: list[dict[str, object]],
) -> dict[str, str]:
    prediction_text_by_qid: dict[str, str] = {}
    for row in answers:
        qid = str(row.get("id") or "").strip()
        if not qid:
            continue
        prediction_text_by_qid[qid] = _cleaned_to_prediction_text(row.get("cleaned", []))
    return prediction_text_by_qid


def _citation_table_from_metrics(
    *,
    model_name: str,
    setup_name: str,
    metrics: dict[str, object],
) -> dict[str, object]:
    return {
        "Model": model_name,
        "Setup": setup_name,
        "CitPr": metrics.get("precision_mean"),
        "CitRec": metrics.get("recall_mean"),
        "CitF1": metrics.get("f1_mean"),
    }


def run_citation_and_refusal_evaluation(
    *,
    generation_file: Path,
    gold_by_qid: dict[str, list[str]],
    expected_refusal_by_qid: dict[str, bool],
    retrieved_by_qid: dict[str, list[str]],
    model_name: str,
    setup_name: str,
    cached_citation_eval: dict[str, Any] | None = None,
) -> dict[str, object]:
    answers = _load_generation_answers(generation_file)
    predicted_by_qid = _load_predicted_ids_by_qid(answers)
    prediction_text_by_qid = _load_prediction_text_by_qid(answers)
    refusals_by_qid = _load_refusal_flags_by_qid(answers)

    if cached_citation_eval is not None:
        citation_result_raw = cached_citation_eval.get("citation_result", {})
        citation_result = citation_result_raw if isinstance(citation_result_raw, dict) else {}
    else:
        citation_result = _evaluate_citations(
            gold_by_qid=gold_by_qid,
            predicted_by_qid=predicted_by_qid,
            refusals_by_qid=refusals_by_qid,
            expected_refusal_by_qid=expected_refusal_by_qid,
        )
    refusal_eval = run_refusal_evaluation(
        gold_by_qid=gold_by_qid,
        expected_refusal_by_qid=expected_refusal_by_qid,
        retrieved_by_qid=retrieved_by_qid,
        predicted_by_qid=predicted_by_qid,
        refusals_by_qid=refusals_by_qid,
        model_name=model_name,
        setup_name=setup_name,
    )
    refusal_result_raw = refusal_eval.get("refusal_result", {})
    refusal_result = refusal_result_raw if isinstance(refusal_result_raw, dict) else {}
    metrics = {
        **citation_result.get("metrics", {}),
        **refusal_eval.get("metrics", {}),
    }
    counts = {
        **citation_result.get("counts", {}),
        **refusal_eval.get("counts", {}),
    }
    metrics["citation_table"] = _citation_table_from_metrics(
        model_name=model_name,
        setup_name=setup_name,
        metrics=metrics,
    )
    return {
        "prediction_text_by_qid": prediction_text_by_qid,
        "predicted_by_qid": predicted_by_qid,
        "predicted_row_count": len(predicted_by_qid),
        "citation_result": citation_result,
        "refusal_result": refusal_result,
        "metrics": metrics,
        "counts": counts,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate generation outputs with citation metrics."
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
        help="Disable citation cache reuse and force recomputation.",
    )
    return parser


def _nan_citation_row(model_name: str, setup_name: str) -> dict[str, object]:
    return {
        "Model": model_name,
        "Setup": setup_name,
        "CitPr": float("nan"),
        "CitRec": float("nan"),
        "CitF1": float("nan"),
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

    citation_rows: list[dict[str, object]] = []
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
            citation_output_file = output_dir / "citation.json"

            if not generation_file.is_file():
                print(
                    f"Missing generation file for setup={setup_name} model={model_name}: "
                    f"returning NaNs (tried: {generation_file})."
                )
                citation_rows.append(_nan_citation_row(model_name, setup_name))
                continue

            cached_citation_eval = None
            if use_cache:
                cached_citation_eval = load_cached_citation_evaluation(
                    citation_output_file=citation_output_file
                )
                if cached_citation_eval is not None:
                    print(
                        "Using cached citation metrics: "
                        f"model={model_name} "
                        f"setup={setup_name} "
                        f"file={citation_output_file}"
                    )

            print(f"Loading predictions from {generation_file}")
            citation_eval = run_citation_and_refusal_evaluation(
                generation_file=generation_file,
                gold_by_qid=gold_by_qid,
                expected_refusal_by_qid=expected_refusal_by_qid,
                retrieved_by_qid=retrieved_by_qid,
                model_name=model_name,
                setup_name=setup_name,
                cached_citation_eval=cached_citation_eval,
            )
            predicted_row_count = int(citation_eval.get("predicted_row_count", 0))
            print(f"Loaded {predicted_row_count} predicted rows.")
            citation_result_raw = citation_eval.get("citation_result", {})
            metrics_raw = citation_eval.get("metrics", {})
            citation_result = citation_result_raw if isinstance(citation_result_raw, dict) else {}
            metrics = metrics_raw if isinstance(metrics_raw, dict) else {}
            print(
                "Citation metrics: "
                f"model={model_name} "
                f"setup={setup_name} "
                f"precision={metrics.get('precision_mean')} "
                f"recall={metrics.get('recall_mean')} "
                f"f1={metrics.get('f1_mean')}"
            )

            citation_payload = build_citation_payload(
                model_name=model_name,
                setup_name=setup_name,
                dataset_id=str(args.dataset_id),
                split=str(args.split),
                lang=lang,
                generation_file=generation_file,
                context_file=context_file,
                citation_result=citation_result,
                metrics=metrics,
            )
            with citation_output_file.open("w", encoding="utf-8") as handle:
                json.dump(citation_payload, handle, ensure_ascii=False, indent=2)
            print(f"Wrote citation evaluation to {citation_output_file}")

            citation_table = metrics.get("citation_table")
            if isinstance(citation_table, dict):
                citation_rows.append(citation_table)
            else:
                citation_rows.append(_nan_citation_row(model_name, setup_name))
            num_completed_runs += 1

    if num_completed_runs == 0:
        print("No real runs were evaluated; all requested rows are NaN.")

    print("All citation rows:")
    _print_rows(citation_rows, CITATION_TABLE_COLUMNS)


if __name__ == "__main__":
    main()
