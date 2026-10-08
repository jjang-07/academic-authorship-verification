"""Evaluation framework for V2 authorship verification.

Public API
----------
Metrics (``evaluation.metrics``)
    evaluate_all, auc_score, c_at_1, f_05_u, f1_metric, brier_score_metric,
    verdict_to_pred_score, debate_results_to_scores, agent_responses_to_scores

Ablation (``evaluation.ablation``)
    run_v1_baseline, run_lip_style, run_naive_mad, run_full_v2, run_all,
    save_result, AblationResult

Topic-stratified (``evaluation.topic_stratified``)
    evaluate_stratified, stratify_ablation_result, compare_v2_vs_v1,
    save_stratified, save_comparison, print_stratified_report, StratifiedResult
"""

from evaluation.ablation import (
    AblationResult,
    run_all,
    run_full_v2,
    run_full_v2_textonly,
    run_lip_style,
    run_naive_mad,
    run_v1_baseline,
    save_result,
)
from evaluation.metrics import (
    agent_responses_to_scores,
    auc_score,
    brier_score_metric,
    c_at_1,
    debate_results_to_scores,
    evaluate_all,
    f_05_u,
    f1_metric,
    verdict_to_pred_score,
)
from evaluation.topic_stratified import (
    StratifiedResult,
    compare_v2_vs_v1,
    evaluate_stratified,
    print_stratified_report,
    save_comparison,
    save_stratified,
    stratify_ablation_result,
)

__all__ = [
    # metrics
    "evaluate_all",
    "auc_score",
    "c_at_1",
    "f_05_u",
    "f1_metric",
    "brier_score_metric",
    "verdict_to_pred_score",
    "debate_results_to_scores",
    "agent_responses_to_scores",
    # ablation
    "AblationResult",
    "run_v1_baseline",
    "run_lip_style",
    "run_naive_mad",
    "run_full_v2",
    "run_full_v2_textonly",
    "run_all",
    "save_result",
    # topic stratified
    "StratifiedResult",
    "evaluate_stratified",
    "stratify_ablation_result",
    "compare_v2_vs_v1",
    "save_stratified",
    "save_comparison",
    "print_stratified_report",
]
