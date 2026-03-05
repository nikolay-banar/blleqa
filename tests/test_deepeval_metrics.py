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


@pytest.mark.skipif(not OPENROUTER_API_KEY_SET, reason="set OPENROUTER_API_KEY for real DeepEval test")
def test_deepeval_correctness_with_multiple_instances() -> None:
    from generation.evaluators.deepeval_metrics import deepeval_correctness

    rows = [
        {
            "id": "case-1",
            "query": "Can a tenant terminate a lease early without penalty in this jurisdiction?",
            "prediction": "A tenant may terminate early only when permitted by statute or contract terms.",
            "ref": "A tenant may terminate early only when permitted by statute or contract terms.",
            "expected_min_score": 3.0,
            "expected_max_score": 5.0,
        },
        {
            "id": "case-2",
            "query": "Is verbal acceptance enforceable for this contract type?",
            "prediction": "Verbal acceptance is always enforceable for every contract type.",
            "ref": "Certain contracts require writing to be enforceable under statute.",
            "expected_min_score": 1.0,
            "expected_max_score": 2.5,
        },
        {
            "id": "case-3",
            "query": "What must an employer prove to dismiss an employee for cause?",
            "prediction": "The employer must show a serious and documented breach that justifies immediate dismissal.",
            "ref": "Dismissal for cause generally requires proof of serious misconduct with supporting evidence.",
            "expected_min_score": 3.0,
            "expected_max_score": 5.0,
        },
        {
            "id": "case-4",
            "query": "Can personal data be processed without consent under GDPR?",
            "prediction": "Yes, processing can rely on legal bases other than consent, such as legal obligation or legitimate interests.",
            "ref": "Under GDPR, consent is one legal basis, but processing may also rely on contract, legal obligation, vital interests, public task, or legitimate interests.",
            "expected_min_score": 3.0,
            "expected_max_score": 5.0,
        },
        {
            "id": "case-5",
            "query": "Is hearsay always inadmissible in civil proceedings?",
            "prediction": "Yes, hearsay is always inadmissible with no exceptions.",
            "ref": "Hearsay may be admissible in some civil contexts, subject to statutory and evidentiary rules.",
            "expected_min_score": 1.0,
            "expected_max_score": 2.5,
        },
        {
            "id": "case-6",
            "query": "When can directors be personally liable for company debts?",
            "prediction": "Directors may face personal liability when they breach duties, trade wrongfully, or commit fraud.",
            "ref": "Personal director liability can arise from breach of fiduciary duties, wrongful trading, fraud, or statutory violations.",
            "expected_min_score": 3.0,
            "expected_max_score": 5.0,
        },
        {
            "id": "case-7",
            "query": "Can a consumer withdraw from an online purchase contract?",
            "prediction": "In many jurisdictions, consumers have a cooling-off period for distance contracts, subject to exceptions.",
            "ref": "Consumer law often grants a withdrawal period for distance contracts, with statutory exceptions for certain goods or services.",
            "expected_min_score": 3.0,
            "expected_max_score": 5.0,
        },
    ]

    model_name = os.getenv("DEEPEVAL_TEST_MODEL", "google/gemma-3-27b-it")
    result = deepeval_correctness(
        rows,
        judge_config={
            "model": model_name,
            "providers": ["novita/bf16"],
        },
    )
    scores = result["scores"]
    failed_eval_ids = result["failed_eval_ids"]
    failure_reasons = result["failure_reasons"]
    evaluation_reasons = result["evaluation_reasons"]

    results_file = Path(
        os.getenv(
            "DEEPEVAL_TEST_RESULTS_FILE",
            "outputs/test_results/deepeval_correctness_results.json",
        )
    )
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "model_name": model_name,
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

    # Evaluation reasons are optional, but if present they must map to known ids and be non-empty.
    assert set(evaluation_reasons.keys()).issubset(expected_ids)
    for reason in evaluation_reasons.values():
        assert isinstance(reason, str)
        assert reason

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
            and expected_ranges[case_id][0] <= score <= expected_ranges[case_id][1]
        )
    ]
    assert not out_of_range, f"Scores outside expected range: {out_of_range}"
