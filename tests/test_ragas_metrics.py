from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

OPENROUTER_API_KEY_SET = bool(os.getenv("OPENROUTER_API_KEY"))


@pytest.mark.skipif(not OPENROUTER_API_KEY_SET, reason="set OPENROUTER_API_KEY for real RAGAS test")
def test_ragas_faithfulness_with_multiple_instances() -> None:
    from generation.evaluators.ragas_metrics import ragas_faithfulness

    rows = [
        {
            "id": "case-1",
            "query": "When was the first Super Bowl?",
            "prediction": "The first Super Bowl was played on January 15, 1967.",
            "contexts": [
                "The First AFL–NFL World Championship Game was played on January 15, 1967."
            ],
            "expected_min_score": 0.7,
            "expected_max_score": 1.0,
        },
        {
            "id": "case-2",
            "query": "When was the first Super Bowl?",
            "prediction": "The first Super Bowl was played in 1972.",
            "contexts": [
                "The First AFL–NFL World Championship Game was played on January 15, 1967."
            ],
            "expected_min_score": 0.0,
            "expected_max_score": 0.4,
        },
        {
            "id": "case-3",
            "query": "What does GDPR require for lawful processing?",
            "prediction": "GDPR allows processing when there is a lawful basis, such as consent or legal obligation.",
            "contexts": [
                "Under GDPR Article 6, processing is lawful when at least one legal basis applies, including consent, contract, legal obligation, vital interests, public task, or legitimate interests."
            ],
            "expected_min_score": 0.6,
            "expected_max_score": 1.0,
        },
        {
            "id": "case-4",
            "query": "What does GDPR require for lawful processing?",
            "prediction": "GDPR only allows processing if the user gives explicit consent.",
            "contexts": [
                "Under GDPR Article 6, processing is lawful when at least one legal basis applies, including consent, contract, legal obligation, vital interests, public task, or legitimate interests."
            ],
            "expected_min_score": 0.0,
            "expected_max_score": 0.5,
        },
    ]

    model_name = os.getenv("RAGAS_TEST_MODEL", "google/gemma-3-27b-it")
    judge_config: dict[str, object] = {
        "model": model_name,
        "providers": ["novita/bf16"],
    }

    result = ragas_faithfulness(rows, judge_config=judge_config)
    scores = result["scores"]
    failed_eval_ids = result["failed_eval_ids"]
    failure_reasons = result["failure_reasons"]

    results_file = Path(
        os.getenv(
            "RAGAS_TEST_RESULTS_FILE",
            "outputs/test_results/ragas_faithfulness_results.json",
        )
    )
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "model_name": model_name,
                "providers": ["novita/bf16"],
                "num_cases": len(rows),
                "result": result,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    expected_ids = {row["id"] for row in rows}
    successful_ids = set(scores.keys())
    failed_ids = set(failed_eval_ids)

    assert successful_ids.isdisjoint(failed_ids)
    assert successful_ids | failed_ids == expected_ids
    assert set(failure_reasons.keys()) == failed_ids
    for failed_id in failed_ids:
        assert isinstance(failure_reasons[failed_id], str)
        assert failure_reasons[failed_id]

    expected_ranges = {
        row["id"]: (row["expected_min_score"], row["expected_max_score"])
        for row in rows
        if row["id"] in scores
    }
    out_of_range = [
        (
            case_id,
            score,
            expected_ranges[case_id][0],
            expected_ranges[case_id][1],
        )
        for case_id, score in scores.items()
        if not (
            isinstance(score, float)
            and 0.0 <= score <= 1.0
            and expected_ranges[case_id][0] <= score <= expected_ranges[case_id][1]
        )
    ]
    assert not out_of_range, f"RAGAS faithfulness scores outside expected range: {out_of_range}"
