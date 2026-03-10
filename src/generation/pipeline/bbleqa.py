import json
from datasets import load_dataset

from .open_router import GenerationInput
from .prompts import SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_TEMPLATE
from ast import literal_eval


def load_blleqa_test_inputs(
    context_file: str,
    dataset_id: str = "clips/bLLeQa_aligned",
    split: str = "test",
    lang: str = 'nl'
) -> list[GenerationInput]:
    dataset = load_dataset(dataset_id, split)[lang]

    inputs: list[GenerationInput] = []

    LANGS = {'nl': "Dutch", 'fr': 'French'}

    with open(context_file, "r", encoding="utf-8") as handle:
        context_by_id = json.load(handle)

    for row in dataset:
        qid = str(row["id"])
        regions = row.get("regions") or []
        if isinstance(regions, str):
            regions = literal_eval(regions)

        topics = row.get("topics") or []
        if isinstance(topics, str):
            topics = topics.split(";")

        user_prompt = USER_PROMPT_TEMPLATE.safe_substitute(
            question=row.get("question", ""),
            regions=", ".join(regions),
            topics="; ".join(topics),
            context=context_by_id[qid],
        )

        system_prompt=SYSTEM_PROMPT_TEMPLATE.safe_substitute(answer_language = LANGS[lang])

        inputs.append(
            {
                "id": qid,
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
            }
        )

    return inputs
