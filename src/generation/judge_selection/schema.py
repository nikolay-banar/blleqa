from generation.evaluators.deepeval_metrics import JudgeConfig


from typing import TypedDict


class JudgeCandidate(TypedDict):
    name: str
    judge_config: JudgeConfig


class JudgeEvaluation(TypedDict, total=False):
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
    mean_pred_by_true_label: dict[str, float]
    num_by_true_label: dict[str, int]
    pearson: float | None
    spearman: float | None
    mae: float | None
    f1_macro: float | None
    f1_macro_per_label: dict[str, float] | None
    f1_macro_t_3: float | None
    f1_macro_per_label_t_3: dict[str, float] | None
    f1_macro_t_4: float | None
    f1_macro_per_label_t_4: dict[str, float] | None
    f1_macro_group_1_2_vs_3_5: float | None
    f1_macro_per_label_group_1_2_vs_3_5: dict[str, float] | None
    f1_macro_group_1_2_vs_3_vs_4_5: float | None
    f1_macro_per_label_group_1_2_vs_3_vs_4_5: dict[str, float] | None
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
