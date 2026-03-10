from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

from generation.pipeline.bbleqa import load_blleqa_test_inputs
from generation.pipeline.open_router import (
    GenerationResult,
    ModelConfig,
    agenerate,
)

DEFAULT_DATASET_ID = "clips/bLLeQa_aligned"
MODEL_CONFIG_DIR = Path(__file__).resolve().parents[1] / "model_configs"

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate answers for the bLLeQa test split using an external context file."
    )
    parser.add_argument(
        "--context-file",
        required=True,
        help="Path to the JSON file containing context keyed by question id.",
    )
    parser.add_argument(
        "--config-dir",
        default=str(MODEL_CONFIG_DIR),
        help="Directory containing model config JSON files.",
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Base directory where generation outputs will be written.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to use for generation.",
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help="Hugging Face dataset id to load.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--lang",
        default="nl",
        choices=("nl", "fr"),
        help="Dataset language.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent generation requests.",
    )
    return parser


async def _amain() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    logger.info("Starting run for model=%s lang=%s split=%s", args.model, args.lang, args.split)
    logger.info("Context file: %s", args.context_file)

    model_config_path = Path(args.config_dir) / f"{args.model}.json"
    with open(model_config_path, "r", encoding="utf-8") as handle:
        loaded_config = json.load(handle)
    if not isinstance(loaded_config, dict):
        raise ValueError("The config file must contain a JSON object.")
    model_config: ModelConfig = dict(loaded_config)

    logger.info("Loaded model config from: %s", model_config_path)

    context_name = Path(args.context_file).stem
    output_path = Path(args.outputs_dir) / args.lang / context_name / f"{args.model}.json"

    logger.info("Output path: %s", output_path)

    previous_output: dict[str, object] | None = None
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as handle:
            loaded_previous_output = json.load(handle)
        if isinstance(loaded_previous_output, dict):
            previous_output = loaded_previous_output

    if previous_output is None:
        logger.info("No previous output found. Running all inputs.")
    else:
        previous_answers = previous_output.get("answers", [])
        previous_count = len(previous_answers) if isinstance(previous_answers, list) else 0
        logger.info("Found previous output with %s completed answers.", previous_count)

    inputs = load_blleqa_test_inputs(
        context_file=args.context_file,
        dataset_id=args.dataset_id,
        split=args.split,
        lang=args.lang,
    )
    total_inputs = len(inputs)
    logger.info("Loaded %s inputs from dataset.", total_inputs)
    if inputs:
        first_input = inputs[0]
        logger.info(
            "First input: id=%s, system_prompt=%r, user_prompt=%r",
            first_input["id"],
            first_input["system_prompt"][:200],
            first_input["user_prompt"][:300],
        )

    if previous_output is not None:
        previous_answers = previous_output.get("answers", [])
        completed_ids = {
            str(item.get("id") or "").strip()
            for item in previous_answers
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        }
        inputs = [item for item in inputs if item["id"] not in completed_ids]
        logger.info(
            "Skipping %s previously completed inputs; %s remaining.",
            total_inputs - len(inputs),
            len(inputs),
        )

    logger.info(
        "Launching generation for %s inputs with max_concurrency=%s.",
        len(inputs),
        args.max_concurrency,
    )
    start_time = time.time()
    answers: list[GenerationResult]
    errors: list[GenerationResult]

    answers, errors = await agenerate(
        inputs=inputs,
        model_config=model_config,
        max_concurrency=args.max_concurrency,
    )
    logger.info(
        "Generation finished with %s answers and %s errors in this run.",
        len(answers),
        len(errors),
    )
    output_payload = {"answers": answers, "errors": errors}
    if previous_output is not None:
        previous_answers = previous_output.get("answers", [])
        if isinstance(previous_answers, list):
            output_payload["answers"] = previous_answers + answers


    output_payload["answers"] = sorted(
        output_payload["answers"],
        key=lambda item: str(item.get("id") or "").strip() if isinstance(item, dict) else "",
    )
    output_payload["errors"] = sorted(
        output_payload["errors"],
        key=lambda item: str(item.get("id") or "").strip() if isinstance(item, dict) else "",
    )

    logger.info("Saving %s total answers to disk.", len(output_payload["answers"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, ensure_ascii=False, indent=2)
    elapsed = time.time() - start_time
    logger.info("Wrote output to %s", output_path)
    logger.info("Done in %.2fs", elapsed)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
