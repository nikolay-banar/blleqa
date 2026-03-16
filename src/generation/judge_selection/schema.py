from generation.evaluators.deepeval_metrics import JudgeConfig


from typing import TypedDict


class JudgeCandidate(TypedDict):
    name: str
    judge_config: JudgeConfig


class JudgeEvaluation(TypedDict):
    name: str
    judge_config: JudgeConfig
    coverage: float
    mean_score: float | None
    score_std: float | None
    failure_rate: float
    num_scored: int
    num_failed: int
    num_compared_with_gold: int
    mean_true_score: float | None
    pearson_correlation: float | None
    spearman_correlation: float | None
    mae: float | None
    f1_macro: float | None
    pearson_correlation_binary: float | None
    spearman_correlation_binary: float | None
    accuracy_binary: float | None
    f1_binary: float | None
    mean_true_binary: float | None
    mean_pred_binary: float | None
    failed_eval_ids: list[str]
    failure_reasons: dict[str, str]


class JudgeSelectionResult(TypedDict):
    selected_judge_name: str
    selected_judge_config: JudgeConfig
    ranking: list[JudgeEvaluation]


class CandidateCorrectnessRun(TypedDict):
    name: str
    judge_config: JudgeConfig
    scores: dict[str, float]
    failed_eval_ids: list[str]
    failure_reasons: dict[str, str]
    evaluation_reasons: dict[str, str]
