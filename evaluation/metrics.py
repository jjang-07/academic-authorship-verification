"""PAN-style authorship verification metrics — ported exactly from V1.

All five metrics follow the PAN AV benchmark convention:
    - Input:  true_y (list/array of 0/1) and pred_scores (float in [0.0, 1.0])
    - Higher is always better.  Brier score is inverted (1 − brier_score_loss)
      so it sits on the same [0, 1] scale as the others.
    - threshold = 0.5 is the "unanswered" / "abstain" boundary used by c@1 and F0.5.

Metric definitions
------------------
AUC       : sklearn roc_auc_score (area under ROC curve)
c@1       : (1/n)(nc + nu × nc/n)
            nc = correctly answered, nu = unanswered (score == 0.5)
            Standard PAN metric — rewards abstaining on genuinely uncertain pairs.
F0.5      : Precision-weighted F with β=0.5.  Uses the PAN "triple-valued"
            binarisation so unanswered predictions neither count as TP nor FP.
F1        : Standard F1 score, filtering out unanswered predictions.
Brier     : 1 − sklearn.metrics.brier_score_loss  (calibration; higher = better)
Overall   : Arithmetic mean of the five metrics above (identical to V1).

All metric values are rounded to 3 decimal places in ``evaluate_all``.

V1 compatibility
-----------------
This module is a direct port of the ``evaluate_all`` family in ``classifier.py``
so results are numerically identical when given the same inputs.

Utilities for LLM-based prediction
------------------------------------
``verdict_to_pred_score(verdict, confidence)``
    Convert an agent's discrete verdict + confidence into a continuous
    same-author probability score suitable for all five metrics.

    SAME_AUTHOR      + conf  →  conf            (e.g. 0.82 → 0.82)
    DIFFERENT_AUTHOR + conf  →  1 − conf         (e.g. 0.75 → 0.25)
    UNKNOWN                  →  0.5 (abstain)
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


# ──────────────────────────────── Helpers ─────────────────────────────────────

def _binarize(
    scores: Sequence[float],
    threshold: float = 0.5,
    triple_valued: bool = False,
) -> np.ndarray:
    """Binarise a score sequence around ``threshold``.

    In triple-valued mode (used by F0.5) scores exactly equal to ``threshold``
    remain as 0.5 (unanswered); in binary mode they are rounded up to 1.
    Mirrors V1's ``binarize()`` exactly.
    """
    y = np.array(scores, dtype=float)
    y = np.ma.fix_invalid(y, fill_value=threshold).data
    if triple_valued:
        y[y > threshold] = 1
    else:
        y[y >= threshold] = 1
    y[y < threshold] = 0
    return y


# ──────────────────────────────── Five Metrics ────────────────────────────────

def auc_score(
    true_y: Sequence[int],
    pred_scores: Sequence[float],
) -> float:
    """Area under the ROC curve.

    Returns 0.0 if only one class is present (degenerate evaluation set).
    """
    try:
        return float(roc_auc_score(list(true_y), list(pred_scores)))
    except ValueError:
        return 0.0


def c_at_1(
    true_y: Sequence[int],
    pred_scores: Sequence[float],
    threshold: float = 0.5,
) -> float:
    """AV-specific metric that rewards abstaining on uncertain cases.

    Formula (PAN standard):
        c@1 = (1/n) × (nc + nu × nc/n)
    where
        nc = number of correctly answered pairs (score ≠ threshold, correct)
        nu = number of unanswered pairs        (score == threshold)
        n  = total number of pairs

    Ported byte-for-byte from V1's ``c_at_1()``.
    """
    n = float(len(pred_scores))
    if n == 0:
        return 0.0
    nc = 0.0
    nu = 0.0
    for gt, score in zip(true_y, pred_scores):
        if score == threshold:
            nu += 1
        elif (score > threshold) == (gt > threshold):
            nc += 1.0
    return (nc + (nc / n) * nu) / n


def f_05_u(
    true_y: Sequence[int],
    pred_scores: Sequence[float],
    pos_label: int = 1,
    threshold: float = 0.5,
) -> float:
    """F0.5 with PAN "unanswered" handling.

    Uses triple-valued binarisation: scores equal to ``threshold`` remain
    unanswered (treated as neither TP, FP, nor FN — they add to the denominator
    only via the 0.25 × n_u term, penalising excessive abstention).

    Denominator: 1.25 × TP + 0.25 × (FN + nu) + FP
    Ported from V1's ``f_05_u_score_metric()``.
    """
    pred_bin = _binarize(pred_scores, threshold=threshold, triple_valued=True)
    n_tp = n_fp = n_fn = n_u = 0
    for p, gt in zip(pred_bin, true_y):
        if p == threshold:
            n_u += 1
        elif p == pos_label and p == gt:
            n_tp += 1
        elif p == pos_label and p != gt:
            n_fp += 1
        elif gt == pos_label and p != gt:
            n_fn += 1
    denom = 1.25 * n_tp + 0.25 * (n_fn + n_u) + n_fp
    return (1.25 * n_tp) / denom if denom > 0 else 0.0


def f1_metric(
    true_y: Sequence[int],
    pred_scores: Sequence[float],
    threshold: float = 0.5,
) -> float:
    """Standard F1 score, filtering out abstained predictions.

    Pairs where pred_score == threshold are excluded from both numerator and
    denominator (they neither help nor hurt).
    Ported from V1's ``f1_metric()``.
    """
    from sklearn.metrics import f1_score

    true_filtered: List[int] = []
    pred_filtered: List[float] = []
    for gt, score in zip(true_y, pred_scores):
        if score != threshold:
            true_filtered.append(gt)
            pred_filtered.append(score)
    if not true_filtered:
        return 0.0
    pred_bin = _binarize(pred_filtered, threshold=threshold)
    return float(f1_score(true_filtered, pred_bin, zero_division=0))


def brier_score_metric(
    true_y: Sequence[int],
    pred_scores: Sequence[float],
) -> float:
    """Inverted Brier score: 1 − sklearn.metrics.brier_score_loss.

    Higher = better calibration.  Range [0, 1] with 1 = perfect.
    Ported from V1's ``brier_score_metric()``.
    """
    try:
        return float(1.0 - brier_score_loss(list(true_y), list(pred_scores)))
    except ValueError:
        return 0.0


# ──────────────────────────────── Combined Evaluator ──────────────────────────

def evaluate_all(
    true_y: Sequence[int],
    pred_scores: Sequence[float],
) -> Dict[str, float]:
    """Compute all five metrics plus the overall mean.

    Returns a dict with keys:
        auc, c@1, f0.5_u, F1, brier, overall

    All values are rounded to 3 decimal places for clean JSON serialisation.
    Matches V1's ``evaluate_all()`` exactly.

    Parameters
    ----------
    true_y      : Binary ground-truth labels (1 = same author, 0 = different).
    pred_scores : Model confidence that the pair is same-author.
                  For LLM-based systems use ``verdict_to_pred_score`` to convert
                  discrete verdicts before calling this function.
    """
    if len(true_y) == 0:
        return {"auc": 0.0, "c@1": 0.0, "f0.5_u": 0.0, "F1": 0.0,
                "brier": 0.0, "overall": 0.0}

    results: Dict[str, float] = {
        "auc":    auc_score(true_y, pred_scores),
        "c@1":    c_at_1(true_y, pred_scores),
        "f0.5_u": f_05_u(true_y, pred_scores),
        "F1":     f1_metric(true_y, pred_scores),
        "brier":  brier_score_metric(true_y, pred_scores),
    }
    results["overall"] = float(np.mean(list(results.values())))
    return {k: round(v, 3) for k, v in results.items()}


def evaluate_extended(
    true_y: Sequence[int],
    pred_scores: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Standard metrics plus same-author accuracy, different-author accuracy, abstention rate.

    Returns
    -------
    Dict with keys: auc, c@1, f0.5_u, F1, brier, overall,
    same_author_accuracy, different_author_accuracy, abstention_rate.
    """
    base = evaluate_all(true_y, pred_scores)
    true_arr = np.array(true_y)
    pred_arr = np.array(pred_scores)

    # Abstention: pred_score == threshold
    abstained = (pred_arr == threshold)
    base["abstention_rate"] = round(float(np.mean(abstained)), 3)

    # Same-author accuracy: of pairs with true_label==1, fraction correct
    same_mask = true_arr == 1
    if np.any(same_mask):
        same_pred = pred_arr[same_mask]
        same_correct = (same_pred > threshold)  # predicted same-author
        base["same_author_accuracy"] = round(float(np.mean(same_correct)), 3)
    else:
        base["same_author_accuracy"] = 0.0

    # Different-author accuracy: of pairs with true_label==0, fraction correct
    diff_mask = true_arr == 0
    if np.any(diff_mask):
        diff_pred = pred_arr[diff_mask]
        diff_correct = (diff_pred < threshold)  # predicted different-author
        base["different_author_accuracy"] = round(float(np.mean(diff_correct)), 3)
    else:
        base["different_author_accuracy"] = 0.0

    return base


# ──────────────────────────────── LLM Score Conversion ───────────────────────

def verdict_to_pred_score(verdict: str, confidence: float) -> float:
    """Map an agent verdict + confidence to a same-author probability score.

    SAME_AUTHOR      + conf → conf
    DIFFERENT_AUTHOR + conf → 1 − conf
    UNKNOWN / other         → 0.5  (treated as abstain by c@1 and F0.5)

    Parameters
    ----------
    verdict    : "SAME_AUTHOR", "DIFFERENT_AUTHOR", or "UNKNOWN".
    confidence : Agent confidence in [0.0, 1.0].
    """
    confidence = max(0.0, min(1.0, float(confidence)))
    v = verdict.upper()
    if v == "SAME_AUTHOR":
        return confidence
    if v == "DIFFERENT_AUTHOR":
        return 1.0 - confidence
    # UNKNOWN or parse failure → abstain
    return 0.5


def debate_results_to_scores(
    pairs,                # List[TextPair]
    debate_results,       # List[DebateResult]
    use_judge: bool = True,
) -> Tuple[List[int], List[float]]:
    """Extract (true_labels, pred_scores) from a list of DebateResult objects.

    Parameters
    ----------
    pairs          : The TextPair list used to generate the results.
    debate_results : Corresponding DebateResult objects.
    use_judge      : If True, use the Judge's final verdict/confidence.
                     If False, use the Analyst's verdict/confidence (LIP-style).

    Returns
    -------
    (true_labels, pred_scores) ready for ``evaluate_all``.
    """
    true_labels: List[int] = []
    pred_scores: List[float] = []
    for pair, result in zip(pairs, debate_results):
        true_labels.append(int(pair.label))
        if use_judge:
            score = verdict_to_pred_score(result.final_verdict, result.final_confidence)
        else:
            score = verdict_to_pred_score(result.analyst.verdict, result.analyst.confidence)
        pred_scores.append(score)
    return true_labels, pred_scores


def agent_responses_to_scores(
    pairs,             # List[TextPair]
    responses,         # List[AgentResponse]
) -> Tuple[List[int], List[float]]:
    """Extract (true_labels, pred_scores) from a list of single-agent responses.

    Used for the LIP-style ablation config (analyst only, no debate).
    """
    true_labels: List[int] = []
    pred_scores: List[float] = []
    for pair, response in zip(pairs, responses):
        true_labels.append(int(pair.label))
        pred_scores.append(
            verdict_to_pred_score(response.verdict, response.confidence)
        )
    return true_labels, pred_scores
