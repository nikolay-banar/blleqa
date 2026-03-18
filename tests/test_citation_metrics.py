from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_citation_score_returns_expected_lists() -> None:
    from generation.evaluators.citation_metrics import citation_score

    y_true = [
        ["doc-a", "doc-b"],
        ["doc-c"],
        [],
        ["doc-x", "doc-x"],
    ]
    y_pred = [
        ["doc-a", "doc-z"],
        [],
        [],
        ["doc-x", "doc-y"],
    ]

    precision, recall, f1 = citation_score(y_true, y_pred)

    assert precision == pytest.approx([0.5, 0.0, 1.0, 0.5], rel=1e-9)
    assert recall == pytest.approx([0.5, 0.0, 1.0, 1.0], rel=1e-9)
    assert f1 == pytest.approx([0.5, 0.0, 1.0, 2 / 3], rel=1e-9)


def test_citation_score_asserts_on_length_mismatch() -> None:
    from generation.evaluators.citation_metrics import citation_score

    with pytest.raises(AssertionError, match="same length"):
        citation_score([["doc-a"]], [["doc-a"], ["doc-b"]])


def test_evaluate_citations_returns_pre_aggregation_and_averages() -> None:
    from generation.evaluators.citation_metrics import _evaluate_citations

    result = _evaluate_citations(
        gold_by_qid={
            "q1": ["doc-a", "doc-b"],
            "q2": ["doc-c"],
        },
        predicted_by_qid={
            "q1": ["doc-a", "doc-z"],
            "q2": ["doc-c"],
        },
        refusals_by_qid={"q1": False, "q2": False},
        expected_refusal_by_qid={"q1": False, "q2": False},
    )

    metrics = result["metrics"]
    assert metrics["unfiltered_precision_mean"] == pytest.approx(0.75, rel=1e-9)
    assert metrics["unfiltered_recall_mean"] == pytest.approx(0.75, rel=1e-9)
    assert metrics["unfiltered_f1_mean"] == pytest.approx(0.75, rel=1e-9)
    assert metrics["precision_mean"] == pytest.approx(0.75, rel=1e-9)
    assert metrics["recall_mean"] == pytest.approx(0.75, rel=1e-9)
    assert metrics["f1_mean"] == pytest.approx(0.75, rel=1e-9)

    q1 = result["by_id"]["q1"]
    q2 = result["by_id"]["q2"]
    assert q1["unfiltered_precision"] == pytest.approx(0.5, rel=1e-9)
    assert q1["unfiltered_recall"] == pytest.approx(0.5, rel=1e-9)
    assert q1["unfiltered_f1"] == pytest.approx(0.5, rel=1e-9)
    assert q1["precision"] == pytest.approx(0.5, rel=1e-9)
    assert q1["recall"] == pytest.approx(0.5, rel=1e-9)
    assert q1["f1"] == pytest.approx(0.5, rel=1e-9)
    assert q2["unfiltered_precision"] == pytest.approx(1.0, rel=1e-9)
    assert q2["unfiltered_recall"] == pytest.approx(1.0, rel=1e-9)
    assert q2["unfiltered_f1"] == pytest.approx(1.0, rel=1e-9)
    assert q2["precision"] == pytest.approx(1.0, rel=1e-9)
    assert q2["recall"] == pytest.approx(1.0, rel=1e-9)
    assert q2["f1"] == pytest.approx(1.0, rel=1e-9)


def test_evaluate_citations_returns_filtered_and_unfiltered_when_refusal_applies() -> None:
    from generation.evaluators.citation_metrics import _evaluate_citations

    result = _evaluate_citations(
        gold_by_qid={"q1": ["doc-a"]},
        predicted_by_qid={"q1": ["doc-z"]},
        refusals_by_qid={"q1": True},
        expected_refusal_by_qid={"q1": True},
    )

    metrics = result["metrics"]
    assert metrics["unfiltered_precision_mean"] == pytest.approx(0.0, rel=1e-9)
    assert metrics["unfiltered_recall_mean"] == pytest.approx(0.0, rel=1e-9)
    assert metrics["unfiltered_f1_mean"] == pytest.approx(0.0, rel=1e-9)
    assert metrics["precision_mean"] == pytest.approx(1.0, rel=1e-9)
    assert metrics["recall_mean"] == pytest.approx(1.0, rel=1e-9)
    assert metrics["f1_mean"] == pytest.approx(1.0, rel=1e-9)

    row = result["by_id"]["q1"]
    assert row["unfiltered_precision"] == pytest.approx(0.0, rel=1e-9)
    assert row["unfiltered_recall"] == pytest.approx(0.0, rel=1e-9)
    assert row["unfiltered_f1"] == pytest.approx(0.0, rel=1e-9)
    assert row["precision"] == pytest.approx(1.0, rel=1e-9)
    assert row["recall"] == pytest.approx(1.0, rel=1e-9)
    assert row["f1"] == pytest.approx(1.0, rel=1e-9)


def test_evaluate_citations_missed_refusal_gets_partial_credit() -> None:
    from generation.evaluators.citation_metrics import _evaluate_citations

    result = _evaluate_citations(
        gold_by_qid={"q1": ["doc-a", "doc-b"]},
        predicted_by_qid={"q1": ["doc-a", "doc-z"]},
        refusals_by_qid={"q1": False},
        expected_refusal_by_qid={"q1": True},
    )

    metrics = result["metrics"]
    assert metrics["unfiltered_precision_mean"] == pytest.approx(0.5, rel=1e-9)
    assert metrics["unfiltered_recall_mean"] == pytest.approx(0.5, rel=1e-9)
    assert metrics["unfiltered_f1_mean"] == pytest.approx(0.5, rel=1e-9)
    assert metrics["precision_mean"] == pytest.approx(0.5, rel=1e-9)
    assert metrics["recall_mean"] == pytest.approx(0.5, rel=1e-9)
    assert metrics["f1_mean"] == pytest.approx(0.5, rel=1e-9)

    row = result["by_id"]["q1"]
    assert row["expected_refusal"] is True
    assert row["predicted_refusal"] is False
    assert row["precision"] == pytest.approx(0.5, rel=1e-9)
    assert row["recall"] == pytest.approx(0.5, rel=1e-9)
    assert row["f1"] == pytest.approx(0.5, rel=1e-9)


def test_evaluate_citations_raises_when_ids_mismatch() -> None:
    from generation.evaluators.citation_metrics import _evaluate_citations

    with pytest.raises(ValueError, match="must contain the same ids"):
        _evaluate_citations(
            gold_by_qid={"q1": ["doc-a"], "q2": ["doc-b"]},
            predicted_by_qid={"q1": ["doc-a"], "q3": ["doc-c"]},
            refusals_by_qid={"q1": False, "q3": False},
            expected_refusal_by_qid={"q1": False, "q2": False},
        )
