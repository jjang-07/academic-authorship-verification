"""Interpretability layer for V2 authorship verification.

Public API
----------
SHAP analysis (``interpretability.shap_analysis``)
    ShapFeature, ShapResult, ShapModel,
    train_shap_model, save_shap_model, load_shap_model,
    explain_pair, explain_pair_from_texts, explain_pairs_batch,
    build_pair_matrix, pair_vector

Report formatting (``interpretability.trace_parser``)
    format_report, save_report, format_and_save, format_batch
"""

from interpretability.shap_analysis import (
    ShapFeature,
    ShapModel,
    ShapResult,
    build_pair_matrix,
    explain_pair,
    explain_pair_from_texts,
    explain_pairs_batch,
    load_shap_model,
    pair_vector,
    save_shap_model,
    train_shap_model,
)
from interpretability.trace_parser import (
    format_and_save,
    format_batch,
    format_report,
    save_report,
)

__all__ = [
    # shap_analysis
    "ShapFeature",
    "ShapResult",
    "ShapModel",
    "train_shap_model",
    "save_shap_model",
    "load_shap_model",
    "explain_pair",
    "explain_pair_from_texts",
    "explain_pairs_batch",
    "build_pair_matrix",
    "pair_vector",
    # trace_parser
    "format_report",
    "save_report",
    "format_and_save",
    "format_batch",
]
