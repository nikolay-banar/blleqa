"""Evaluation utilities."""

from typing import Any


def deepeval_correctness(*args: Any, **kwargs: Any) -> Any:
    from .deepeval_metrics import deepeval_correctness as _deepeval_correctness

    return _deepeval_correctness(*args, **kwargs)


def ragas_faithfulness(*args: Any, **kwargs: Any) -> Any:
    from .ragas_metrics import ragas_faithfulness as _ragas_faithfulness

    return _ragas_faithfulness(*args, **kwargs)

__all__ = [
    "deepeval_correctness",
    "ragas_faithfulness",
]
