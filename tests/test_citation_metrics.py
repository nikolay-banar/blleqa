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
