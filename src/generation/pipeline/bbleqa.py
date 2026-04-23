import json
import logging
from ast import literal_eval

from datasets import load_dataset

from .open_router import GenerationInput
from .prompts import SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


def _normalize_context_map(value: object) -> dict[str, str]:
    # Expected input shape from build_context output:
    # [{"id": "...", "text": "..."}, ...]
    if not isinstance(value, list):
        return {}

    context_map: dict[str, str] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        article_id = str(item.get("id") or "").strip()
        text = str(item.get("text") or "").strip()
        if article_id and text:
            context_map[article_id] = text
    return context_map


def load_blleqa_test_inputs(
    context_file: str,
    dataset_id: str = "clips/bLLeQA",
    split: str = "test",
    lang: str = "nl",
) -> list[GenerationInput]:
    dataset = load_dataset(dataset_id, split)[lang]
    inputs: list[GenerationInput] = []
    langs = {"nl": "Dutch", "fr": "French"}

    with open(context_file, "r", encoding="utf-8") as handle:
        loaded_context = json.load(handle)
    if not isinstance(loaded_context, dict):
        raise ValueError("Context file must be a JSON object keyed by question id.")

    context_by_id = {
        str(qid).strip(): _normalize_context_map(payload)
        for qid, payload in loaded_context.items()
        if str(qid).strip()
    }

    for row in dataset:
        qid = str(row["id"]).strip()
        regions = row.get("regions") or []
        if isinstance(regions, str):
            try:
                regions = literal_eval(regions)
            except (ValueError, SyntaxError):
                regions = [regions]

        topics = row.get("topics") or []
        if isinstance(topics, str):
            topics = [topic.strip() for topic in topics.split(";") if topic.strip()]

        context_map = context_by_id.get(qid, {})
        if not context_map:
            logger.warning("No context found for qid=%s in %s; using empty context.", qid, context_file)

        user_prompt = USER_PROMPT_TEMPLATE.safe_substitute(
            question=row.get("question", ""),
            regions=", ".join(str(region) for region in regions),
            topics="; ".join(str(topic) for topic in topics),
            context=json.dumps(context_map, ensure_ascii=False),
        )
        system_prompt = SYSTEM_PROMPT_TEMPLATE.safe_substitute(
            answer_language=langs.get(lang, "Dutch")
        )

        inputs.append(
            {
                "id": qid,
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
            }
        )

    return inputs


def _to_id_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        return [item for item in stripped.split() if item]
    return [str(value).strip()] if str(value).strip() else []


def _load_gold_ids_by_qid(
    *,
    dataset_id: str,
    split: str,
    lang: str,
) -> dict[str, list[str]]:

    dataset = load_dataset(dataset_id, split)[lang]
    gold_by_qid: dict[str, list[str]] = {}
    for row in dataset:
        qid = str(row.get("id") or "").strip()
        if not qid:
            continue
        gold_by_qid[qid] = _to_id_list(row.get("article_ids"))
    return gold_by_qid


def _to_text(value: object, *, lang: str | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_to_text(item, lang=lang) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        if lang and lang in value:
            lang_text = _to_text(value.get(lang), lang=lang)
            if lang_text:
                return lang_text
        for key in ("text", "answer", "content", "value"):
            if key in value:
                nested_text = _to_text(value.get(key), lang=lang)
                if nested_text:
                    return nested_text
        parts = [_to_text(item, lang=lang) for item in value.values()]
        return "\n".join(part for part in parts if part).strip()
    return str(value).strip()


def _extract_query_text(row: dict[str, object], *, lang: str) -> str:
    for key in ("question", "query", "questions", "prompt"):
        if key in row:
            text = _to_text(row.get(key), lang=lang)
            if text:
                return text
    return ""


def _extract_reference_text(row: dict[str, object], *, lang: str) -> str:
    for key in (
        "answer",
        "answers",
        "reference",
        "reference_answer",
        "gold_answer",
        "gold_answers",
        "ref",
        "response",
    ):
        if key in row:
            text = _to_text(row.get(key), lang=lang)
            if text:
                return text
    return ""


def _load_gold_query_ref_by_qid(
    *,
    dataset_id: str,
    split: str,
    lang: str,
) -> tuple[dict[str, str], dict[str, str]]:

    dataset = load_dataset(dataset_id, split)[lang]
    query_by_qid: dict[str, str] = {}
    ref_by_qid: dict[str, str] = {}
    for row in dataset:
        qid = str(row.get("id") or "").strip()
        if not qid:
            continue
        query_by_qid[qid] = _extract_query_text(row, lang=lang)
        ref_by_qid[qid] = _extract_reference_text(row, lang=lang)
    return query_by_qid, ref_by_qid
