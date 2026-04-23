from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any
from ast import literal_eval
from datasets import load_dataset
from transformers import AutoTokenizer

DEFAULT_DATASET_ID = "clips/bLLeQA"
DEFAULT_RETRIEVED_DIR = Path("data/retrieved")
DEFAULT_OUTPUT_DIR = Path("data/context")
TOP_K = 100

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-question context JSON and retrieval metrics from retrieved article ids. "
            "Context outputs contain lists of {'id', 'text'} article entries per question."
        )
    )
    parser.add_argument(
        "--retrieved-file",
        type=Path,
        required=True,
        help=(
            "Path to retrieved docs JSON keyed by question id: "
            "{qid: [article_id, ...]}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Base output directory. If omitted, writes under data/context. "
            "Writes metrics to <output-dir>/<retrieved-file-stem>_metrics.json and "
            "contexts to <output-dir>/<lang>/<retrieved-file-stem>_<type>.json."
        ),
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split/config passed to load_dataset(dataset_id, split).",
    )
    parser.add_argument(
        "--lang",
        default="nl",
        choices=("nl", "fr"),
        help=(
            "Language used to compute top_100/top_100_plus_gold ids and metrics. "
            "Context files are still generated for both nl and fr."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    return parser


def _to_ref_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, dict):
        return [str(item).strip() for item in value.keys() if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                parsed = literal_eval(stripped)
            except (ValueError, SyntaxError):
                return [stripped]
            return _to_ref_list(parsed)
        return [stripped]
    return [str(value).strip()] if str(value).strip() else []


def _build_context_ids(
    *,
    questions: list[dict[str, Any]],
    retrieved_docs: dict[str, list[str]],
    top_k: int = TOP_K,
) -> dict[str, Any]:

    only_gold: dict[str, list[str]] = {}
    top_100: dict[str, list[str]] = {}
    top_100_plus_gold: dict[str, list[str]] = {}


    skipped_invalid = 0
    skipped_out_of_range = 0
    injected_gold = 0
    recall = []
    hits = []
    gold_recall = []
    gold_hits = []


    for question in questions:
        qid = str(question.get("id") or "").strip()
        if not qid:
            continue
        
        gold_refs = question["article_ids"].split()
        only_gold[qid] = gold_refs

        retrieved = _to_ref_list(retrieved_docs.get(qid, []))[:top_k]
        top_100[qid] = retrieved

        if gold_refs:
            r = len(set(retrieved).intersection(set(gold_refs))) / len(gold_refs)
            h = 1.0 if r == 1 else 0
        else:
            r = 0.0
            h = 0.0
        recall.append(r)
        hits.append(h)

        rng = random.Random(qid)
        missing_gold = [ref for ref in gold_refs if ref not in retrieved]
        rng.shuffle(missing_gold)
        retrieved_plus_gold = list(retrieved)
        if missing_gold:
            replace_candidates = [i for i, item in enumerate(retrieved) if item not in gold_refs]
            replacement_count = min(len(missing_gold), len(replace_candidates))
            if replacement_count < len(missing_gold):
                logger.warning(
                    "Could only inject %s/%s missing gold references for %s.",
                    replacement_count,
                    len(missing_gold),
                    qid,
                )
            sampled_positions = rng.sample(replace_candidates, k=replacement_count)

            for position, gold_ref in zip(sampled_positions, missing_gold[:replacement_count]):
                retrieved_plus_gold[position] = gold_ref
                injected_gold += 1

        if len(retrieved) > top_k:
            raise ValueError(f"top_100 for {qid} exceeds top_k={top_k}: {len(retrieved)}")
        if len(retrieved_plus_gold) > top_k:
            raise ValueError(f"top_100_plus_gold for {qid} exceeds top_k={top_k}: {len(retrieved_plus_gold)}")
        if len(retrieved_plus_gold) != len(retrieved):
            raise ValueError(
                f"top_100_plus_gold length mismatch for {qid}: {len(retrieved_plus_gold)} != {len(retrieved)}"
            )

        top_100_plus_gold[qid] = retrieved_plus_gold

        if gold_refs:
            gr = len(set(retrieved_plus_gold).intersection(set(gold_refs))) / len(gold_refs)
            gh = 1.0 if gr == 1 else 0
        else:
            gr = 0.0
            gh = 0.0
        gold_recall.append(gr)
        gold_hits.append(gh)


    if skipped_invalid:
        logger.warning("Skipped %s non-integer retrieved indices.", skipped_invalid)
    if skipped_out_of_range:
        logger.warning("Skipped %s out-of-range retrieved indices.", skipped_out_of_range)
    if injected_gold:
        logger.info("Injected %s missing gold references into top_100_plus_gold.", injected_gold)
    if recall:
        logger.info("Average recall: %.4f", sum(recall) / len(recall))
    if hits:
        logger.info("Average hit rate: %.4f", sum(hits) / len(hits))

    return {
        "context": {
            "only_gold": only_gold,
            "top_100": top_100,
            "top_100_plus_gold": top_100_plus_gold
            },

        "recall": {
            "only_gold": 1,
            "top_100":  sum(recall)/len(recall),
            "top_100_plus_gold":  sum(gold_recall)/len(gold_recall)
            },

        "hits": {
            "only_gold": 1,
            "top_100": sum(hits)/len(hits),
            "top_100_plus_gold":  sum(gold_hits)/len(gold_hits)
            }
        }

def _build_context(
    *,
    questions: list[dict[str, Any]],
    context_ids: dict[str, list[str]],
    corpus,
    tokenizer=None,
) -> dict[str, dict[str, dict[str, str]]]:
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
    corpus_by_id : dict[str, str] = {}
    corpus_len : dict[str, float] = {}
    context_by_qid = {}

    for row in corpus:
        text = f"{row['reference']}\n{row['article']}"
        corpus_by_id[str(row['id'])] = text
        corpus_len[str(row['id'])] = len(tokenizer(text)["input_ids"])

    for question in questions:
        qid = str(question.get("id") or "").strip()
        if not qid:
            continue

        if len(context_ids[qid]) > TOP_K:
            raise ValueError(f"Context for {qid} exceeds TOP_K={TOP_K}: {len(context_ids[qid])}")

        context = [ 
            {"id": ref, "text": corpus_by_id[ref]} for ref in context_ids[qid] if ref in corpus_by_id and corpus_by_id[ref]
        ]
        context_by_qid[qid] = context
        context_len = sum([corpus_len[ref] for ref in context_ids[qid] if ref in corpus_len and corpus_len[ref]])

        if context_len > 90000:
            logger.warning("Extra large context for %s: %s tokens", qid, context_len)

    return context_by_qid

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    parser = _build_arg_parser()
    args = parser.parse_args()

    retrieved_file = args.retrieved_file
    if not retrieved_file.is_file():
        candidate = DEFAULT_RETRIEVED_DIR / retrieved_file
        if candidate.is_file():
            retrieved_file = candidate
        else:
            raise FileNotFoundError(f"Retrieved file not found: {args.retrieved_file}")

    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    output_prefix = retrieved_file.stem

    logger.info("Reading retrieved indices from: %s", retrieved_file)

    with open(retrieved_file, "r", encoding="utf-8") as handle:
        retrieved_docs = json.load(handle)

    if not isinstance(retrieved_docs, dict):
        raise ValueError("Retrieved file must be a JSON object keyed by question id.")
    
    logger.info("Loaded retrieved docs for %s question ids.", len(retrieved_docs))

    questions = load_dataset(args.dataset_id, args.split)

    logger.info(
        "Loaded %s Dutch questions and %s French questions.",
        len(questions["nl"]),
        len(questions["fr"]),
    )

    logger.info("Loading article corpus from dataset_id=%s", args.dataset_id)

    context_by_qid = _build_context_ids(
        questions=list(questions[args.lang]),
        retrieved_docs=retrieved_docs,
        top_k=TOP_K,
    )
    logger.info("Built %s context types.", len(context_by_qid))

    context_payload = context_by_qid["context"]
    built_context_by_lang = {}
    for lang in ["nl", "fr"]:
        corpus = load_dataset(args.dataset_id, "corpus")[lang]
        built_context_by_lang[lang] = {}
        for context_type, cont in context_payload.items():
            built_context_by_lang[lang][context_type] = _build_context(
                questions=list(questions[lang]),
                context_ids=cont,
                corpus=corpus,
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"{output_prefix}_metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {metrics_path} (use --overwrite)")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "recall": context_by_qid["recall"],
                "hits": context_by_qid["hits"],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Wrote metrics JSON to: %s", metrics_path)

    for lang, context_by_type in built_context_by_lang.items():
        lang_output_dir = output_dir / lang
        lang_output_dir.mkdir(parents=True, exist_ok=True)
        for context_type, payload in context_by_type.items():
            ordered_payload = {qid: payload[qid] for qid in sorted(payload)}
            path = lang_output_dir / f"{output_prefix}_{context_type}.json"
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"Output file already exists: {path} (use --overwrite)")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(ordered_payload, handle, ensure_ascii=False, indent=2)
            logger.info("Wrote %s context JSON for %s to: %s", context_type, lang, path)


if __name__ == "__main__":
    main()
