from __future__ import annotations

import asyncio
import json
import os
from typing import Any, TypedDict
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_DATASET_ID = "clips/bLLeQa_aligned"


class ModelConfig(TypedDict, total=False):
    name: str
    url: str
    base_url: str
    api_key: str
    providers: list[str]
    max_completion_tokens: int



class GenerationInput(TypedDict):
    id: str
    system_prompt: str
    user_prompt: str


class QAChunk(TypedDict):
    text: str
    supported_sources: list[str]


class GenerationResult(TypedDict, total=False):
    id: str
    raw: dict[str, object]
    cleaned: list[QAChunk]
    error: str


QA_SCHEMA = {
    "type": "object",
    "properties": {
        "qa_chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "supported_sources": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["text", "supported_sources"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["qa_chunks"],
    "additionalProperties": False,
}


def _extract_json_string(content: str) -> str:
    stripped = content.strip()
    if not stripped:
        return '{"qa_chunks": []}'

    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start != -1 and end != -1 and end >= start:
            return stripped[start : end + 1]

    return stripped


def _parse_content(content: str) -> object:
    try:
        return json.loads(_extract_json_string(content))
    except json.JSONDecodeError as exc:
        raise ValueError("Model response was not valid JSON.") from exc


def _normalize_chunks(payload: object) -> list[QAChunk]:
    if not isinstance(payload, dict):
        raise ValueError("Expected the model response to be a JSON object.")

    raw_chunks = payload.get("qa_chunks")
    if not isinstance(raw_chunks, list):
        raise ValueError("Expected the model response to include a qa_chunks list.")

    cleaned: list[QAChunk] = []
    for chunk in raw_chunks:
        if not isinstance(chunk, dict):
            continue

        text = str(chunk.get("text", "")).strip()
        raw_sources = chunk.get("supported_sources", [])
        if isinstance(raw_sources, list):
            supported_sources = [
                str(source).strip() for source in raw_sources if str(source).strip()
            ]
        else:
            supported_sources = []

        cleaned.append(
            {
                "text": text,
                "supported_sources": supported_sources,
            }
        )

    return cleaned


def _response_to_dict(response: object) -> dict[str, object]:
    if hasattr(response, "model_dump"):
        return dict(response.model_dump())  # type: ignore[call-arg]
    if isinstance(response, dict):
        return response
    return {"response": str(response)}


def _get_client(model_config: ModelConfig | None = None) -> AsyncOpenAI:
    config = model_config or {}

    if not config.get("name", ""):
        raise ValueError("The model field is missing from model_config")

    base_url = str(
        config.get("base_url") or config.get("url") or DEFAULT_OPENROUTER_URL
    ).strip()
    if not base_url:
        raise ValueError("A valid base URL is required.")

    api_key = str(
        config.get("api_key") or os.getenv("OPENROUTER_API_KEY") or ""
    ).strip()
    if not api_key:
        raise ValueError(
            "OpenRouter API key is required. Set model_config['api_key'] or OPENROUTER_API_KEY."
        )

    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def _build_extra_body(model_config: ModelConfig) -> dict[str, object]:
    provider_only = model_config.get("providers")

    extra_body: dict[str, object] = {
        "plugins": [{"id": "response-healing"}], 
        "reasoning": {"effort": "medium"},
        }
    
    provider_payload: dict[str, object] = {"data_collection": "deny"}
    
    if provider_only is not None:
        provider_payload["only"] = provider_only
    
    extra_body["provider"] = provider_payload
    
    return extra_body


async def _generate_one(
    *,
    client: AsyncOpenAI,
    model_config: ModelConfig,
    gen_input: GenerationInput,
    extra_body: dict[str, object],
) -> GenerationResult:

    request_kwargs: dict[str, Any] = {
        "model":model_config["name"],
        "messages": [
            {"role": "system", "content": gen_input["system_prompt"]},
            {"role": "user", "content": gen_input["user_prompt"]},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "qa_chunks",
                "strict": True,
                "schema": QA_SCHEMA,
            },
        },
        "extra_body": extra_body
        }
    
    max_completion_tokens = model_config.get("max_completion_tokens")
    if max_completion_tokens is not None:
        request_kwargs["max_completion_tokens"] = int(max_completion_tokens)

    response = await client.chat.completions.create(**request_kwargs)

    raw_response = _response_to_dict(response)
    message = response.choices[0].message
    content = message.content
    if not content:
        reasoning = getattr(message, "reasoning", None)
        content = str(reasoning).strip() if reasoning else '{"qa_chunks": []}'

    cleaned = _normalize_chunks(_parse_content(content))

    if not cleaned:
        raise ValueError(f"The final output is empty for {gen_input['id']}")

    return {
        "id": gen_input["id"],
        "raw": raw_response,
        "cleaned": cleaned,
    }


async def agenerate(
    *,
    inputs: list[GenerationInput],
    model_config: ModelConfig,
    max_concurrency: int = 5,
) -> tuple[list[GenerationResult], list[GenerationResult]]:
    if not inputs:
        return [], []

    concurrency = max(1, max_concurrency)
    semaphore = asyncio.Semaphore(concurrency)
    client = _get_client(model_config)
    extra_body = _build_extra_body(model_config)

    max_retries = max(3, int(os.getenv("OPENROUTER_MAX_RETRIES", "3") or "0"))
    base_delay = float(os.getenv("OPENROUTER_RETRY_BASE_DELAY", "1.0") or "1.0")
    max_delay = float(os.getenv("OPENROUTER_RETRY_MAX_DELAY", "10.0") or "10.0")

    async def _run_one(item: GenerationInput) -> GenerationResult:
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                async with semaphore:
                    return await _generate_one(
                        client=client,
                        model_config=model_config,
                        gen_input=item,
                        extra_body=extra_body,
                    )
            except Exception as exc:
                last_exc = exc
                if attempt >= max_retries:
                    break

                delay = min(max_delay, base_delay * (2 ** attempt))
                await asyncio.sleep(delay)

        assert last_exc is not None
        raise last_exc
    
    results: list[GenerationResult] = []
    errors: list[GenerationResult] = []

    async def _run_one_captured(
        item: GenerationInput,
    ) -> tuple[GenerationInput, GenerationResult | None, Exception | None]:
        try:
            return item, await _run_one(item), None
        except Exception as exc:
            return item, None, exc

    tasks = [asyncio.create_task(_run_one_captured(item)) for item in inputs]
    for task in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Generating",
    ):
        item, result, exc = await task
        if exc is not None:
            errors.append(
                {
                    "id": item["id"],
                    "raw": {},
                    "cleaned": [],
                    "error": str(exc),
                }
            )
            continue
        if result is not None:
            results.append(result)
    return results, errors
