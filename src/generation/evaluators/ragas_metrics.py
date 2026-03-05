import os
import math
from typing import TypedDict

from datasets import Dataset as HFDataset  # type: ignore
from openai import AsyncOpenAI
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness


class FaithfulnessInput(TypedDict):
    id: str
    query: str
    prediction: str
    contexts: list[str]


class FaithfulnessResult(TypedDict):
    scores: dict[str, float]
    failed_eval_ids: list[str]
    failure_reasons: dict[str, str]


class RagasJudgeConfig(TypedDict, total=False):
    llm: object
    model: str
    url: str
    api_key: str
    providers: list[str]
    batch_size: int


def _to_contexts(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []



def _build_faithfulness_scorer(judge_config: RagasJudgeConfig | None) -> Faithfulness:
    if judge_config and "llm" in judge_config and judge_config["llm"] is not None:
        return Faithfulness(llm=judge_config["llm"])

    model: str | None = None
    url = "https://openrouter.ai/api/v1"
    api_key = os.getenv("OPENROUTER_API_KEY")
    providers: list[str] | None = None
    if judge_config:
        model = judge_config.get("model", model)
        url = judge_config.get("url", url)
        api_key = judge_config.get("api_key", api_key)
        providers = judge_config.get("providers", providers)

    if model is None or not str(model).strip():
        raise ValueError("Model is required. Set judge_config['model'].")
    if api_key is None or not str(api_key).strip():
        raise ValueError(
            "OpenRouter API key is required. Set judge_config['api_key'] or OPENROUTER_API_KEY."
        )

    client_kwargs: dict[str, object] = {"base_url": url, "api_key": api_key}
    client: object = AsyncOpenAI(**client_kwargs)

    llm_kwargs: dict[str, object] = {"client": client}
    if providers:
        llm_kwargs["extra_body"] = {"provider": {"only": providers}}

    llm = llm_factory(model, **llm_kwargs)
    return Faithfulness(llm=llm)

def ragas_faithfulness(
    input_rows: list[FaithfulnessInput],
    judge_config: RagasJudgeConfig | None = None,
) -> FaithfulnessResult:
    if not input_rows:
        return {
            "scores": {},
            "failed_eval_ids": [],
            "failure_reasons": {},
        }

    faithfulness_metric = _build_faithfulness_scorer(judge_config)
    batch_size = 5
    if judge_config and "batch_size" in judge_config:
        batch_size = int(judge_config["batch_size"])

    ids = [row["id"] for row in input_rows]
    eval_dataset = HFDataset.from_dict(
        {
            "user_input": [row["query"] for row in input_rows],
            "response": [row["prediction"] for row in input_rows],
            "retrieved_contexts": [_to_contexts(row["contexts"]) for row in input_rows],
        }
    )

    evaluate_kwargs: dict[str, object] = {
        "dataset": eval_dataset,
        "metrics": [faithfulness_metric],
        "batch_size": batch_size,
        "raise_exceptions": False,
        "show_progress": True,
    }

    result = evaluate(**evaluate_kwargs)
    score_rows = result.to_pandas()  # type: ignore[attr-defined]

    print(score_rows)

    scores: dict[str, float] = {}
    failed_eval_ids: list[str] = []
    failure_reasons: dict[str, str] = {}

    for i, case_id in enumerate(ids):
        row_score = None
        if i < len(score_rows):
            row_score = score_rows.iloc[i].get("faithfulness")

        if row_score is None:
            failed_eval_ids.append(case_id)
            failure_reasons[case_id] = "missing_faithfulness_score"
            continue
        try:
            numeric_score = float(row_score)
        except (TypeError, ValueError):
            failed_eval_ids.append(case_id)
            failure_reasons[case_id] = "missing_faithfulness_score"
            continue
        if math.isnan(numeric_score):
            failed_eval_ids.append(case_id)
            failure_reasons[case_id] = "missing_faithfulness_score"
            continue
        scores[case_id] = numeric_score

    return {
        "scores": scores,
        "failed_eval_ids": failed_eval_ids,
        "failure_reasons": failure_reasons,
    }
