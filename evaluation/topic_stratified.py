"""Topic-stratified authorship verification evaluation.

V1's known weakness is cross-topic pairs: the Gradient Boosting classifier
trained on Reuters news (homogeneous domain) performed worse when text A and
text B covered different subjects.  This module directly measures that gap and
quantifies V2's improvement.

Stratification logic
---------------------
A pair is "same-topic" when ``pair.topic_a.lower() == pair.topic_b.lower()``
and both fields are non-empty.

A pair is "cross-topic" when the normalised topic strings differ and both are
non-empty.

A pair is "unknown-topic" when either topic field is empty (e.g., Reuters-50-50
which carries no explicit topic tags).  Unknown-topic pairs form their own
stratum so they neither inflate nor deflate the two main strata.

Usage
------
    from evaluation.topic_stratified import evaluate_stratified, save_stratified

    # After running an ablation config:
    result = evaluate_stratified(
        pairs, true_labels, pred_scores,
        config_name="FULL_V2",
        v1_cross_topic_overall=0.641,   # from V1 experiments
    )
    print(result.delta_cross_topic)
    save_stratified(result)

V2 vs V1 delta
--------------
If ``v1_cross_topic_overall`` (and/or ``v1_same_topic_overall``) is supplied,
the module computes the signed delta:

    delta = V2_overall − V1_overall

Positive delta = V2 improves over V1 in that stratum.
This directly answers the core science-fair question:
"Does the multi-agent debate help more on cross-topic pairs?"

Output format
--------------
All results are serialised to JSON in ``experiments/results/`` so they can be
loaded for the final write-up tables.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data.preprocessor import TextPair
from evaluation.metrics import evaluate_all


# ──────────────────────────────── Strata Constants ────────────────────────────

SAME_TOPIC    = "same_topic"
CROSS_TOPIC   = "cross_topic"
UNKNOWN_TOPIC = "unknown_topic"


# ──────────────────────────────── Result Dataclass ────────────────────────────

@dataclass
class StratifiedResult:
    """Metrics broken down by topic stratum, plus V2-vs-V1 deltas.

    Fields
    ------
    config_name              : Ablation config that produced the predictions.
    all_metrics              : Metrics computed on the full pair set.
    same_topic_metrics       : Metrics for same-topic pairs only.
    cross_topic_metrics      : Metrics for cross-topic pairs only.
    unknown_topic_metrics    : Metrics for pairs with missing topic tags.
    n_all                    : Total pairs.
    n_same_topic             : Same-topic pair count.
    n_cross_topic            : Cross-topic pair count.
    n_unknown_topic          : Unknown-topic pair count.
    delta_cross_topic        : V2 overall − V1 cross-topic overall (None if
                               V1 baseline not provided).
    delta_same_topic         : V2 overall − V1 same-topic overall (None if
                               V1 baseline not provided).
    v1_cross_topic_overall   : Supplied V1 cross-topic overall score for delta.
    v1_same_topic_overall    : Supplied V1 same-topic overall score for delta.
    timestamp                : ISO-8601 UTC timestamp.
    per_pair                 : Per-pair metadata with stratum tag for audit.
    """

    config_name: str
    all_metrics: Dict[str, float]
    same_topic_metrics: Dict[str, float]
    cross_topic_metrics: Dict[str, float]
    unknown_topic_metrics: Dict[str, float]
    n_all: int
    n_same_topic: int
    n_cross_topic: int
    n_unknown_topic: int
    delta_cross_topic: Optional[float]
    delta_same_topic: Optional[float]
    v1_cross_topic_overall: Optional[float]
    v1_same_topic_overall: Optional[float]
    timestamp: str
    per_pair: List[Dict[str, Any]] = field(default_factory=list)


# ──────────────────────────────── Stratification ─────────────────────────────

def _topic_stratum(pair: TextPair) -> str:
    """Assign a pair to SAME_TOPIC, CROSS_TOPIC, or UNKNOWN_TOPIC."""
    ta = (pair.topic_a or "").strip().lower()
    tb = (pair.topic_b or "").strip().lower()
    if not ta or not tb:
        return UNKNOWN_TOPIC
    return SAME_TOPIC if ta == tb else CROSS_TOPIC


def stratify_pairs(
    pairs: List[TextPair],
) -> Dict[str, List[Tuple[int, TextPair, float]]]:
    """Group pairs by topic stratum without predictions.

    Returns
    -------
    Dict mapping stratum name → list of (original_index, pair).
    """
    groups: Dict[str, List[Tuple[int, TextPair]]] = {
        SAME_TOPIC: [], CROSS_TOPIC: [], UNKNOWN_TOPIC: []
    }
    for idx, pair in enumerate(pairs):
        groups[_topic_stratum(pair)].append((idx, pair))
    return groups  # type: ignore[return-value]


# ──────────────────────────────── Main Evaluator ──────────────────────────────

def evaluate_stratified(
    pairs: List[TextPair],
    true_labels: List[int],
    pred_scores: List[float],
    config_name: str,
    v1_cross_topic_overall: Optional[float] = None,
    v1_same_topic_overall: Optional[float] = None,
) -> StratifiedResult:
    """Compute metrics for each topic stratum.

    Parameters
    ----------
    pairs                    : Original TextPair list.
    true_labels              : Ground-truth labels (1 = same author, 0 = different).
    pred_scores              : Same-author probability scores in [0, 1].
    config_name              : Name of the ablation config (for JSON labelling).
    v1_cross_topic_overall   : Optional V1 cross-topic overall score for delta.
    v1_same_topic_overall    : Optional V1 same-topic overall score for delta.

    Returns
    -------
    StratifiedResult
    """
    if len(pairs) != len(true_labels) or len(pairs) != len(pred_scores):
        raise ValueError(
            f"Length mismatch: pairs={len(pairs)}, "
            f"true_labels={len(true_labels)}, pred_scores={len(pred_scores)}"
        )

    # Assign strata
    strata = [_topic_stratum(p) for p in pairs]

    # Per-stratum accumulators
    stratum_true:  Dict[str, List[int]]   = {SAME_TOPIC: [], CROSS_TOPIC: [], UNKNOWN_TOPIC: []}
    stratum_pred:  Dict[str, List[float]] = {SAME_TOPIC: [], CROSS_TOPIC: [], UNKNOWN_TOPIC: []}
    per_pair: List[Dict[str, Any]] = []

    for i, (pair, stratum, gt, pred) in enumerate(
        zip(pairs, strata, true_labels, pred_scores)
    ):
        stratum_true[stratum].append(gt)
        stratum_pred[stratum].append(pred)
        per_pair.append({
            "pair_id":    i,
            "true_label": gt,
            "pred_score": round(pred, 4),
            "topic_a":    pair.topic_a or "",
            "topic_b":    pair.topic_b or "",
            "stratum":    stratum,
            "author_id":  pair.author_id or "",
            "dataset":    pair.source_dataset or "",
        })

    # Evaluate each stratum
    all_metrics     = evaluate_all(true_labels, pred_scores)
    same_metrics    = evaluate_all(stratum_true[SAME_TOPIC],    stratum_pred[SAME_TOPIC])
    cross_metrics   = evaluate_all(stratum_true[CROSS_TOPIC],   stratum_pred[CROSS_TOPIC])
    unknown_metrics = evaluate_all(stratum_true[UNKNOWN_TOPIC], stratum_pred[UNKNOWN_TOPIC])

    # V2 vs V1 deltas
    delta_cross = (
        round(cross_metrics["overall"] - v1_cross_topic_overall, 3)
        if v1_cross_topic_overall is not None and stratum_true[CROSS_TOPIC]
        else None
    )
    delta_same = (
        round(same_metrics["overall"] - v1_same_topic_overall, 3)
        if v1_same_topic_overall is not None and stratum_true[SAME_TOPIC]
        else None
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    result = StratifiedResult(
        config_name=config_name,
        all_metrics=all_metrics,
        same_topic_metrics=same_metrics,
        cross_topic_metrics=cross_metrics,
        unknown_topic_metrics=unknown_metrics,
        n_all=len(pairs),
        n_same_topic=len(stratum_true[SAME_TOPIC]),
        n_cross_topic=len(stratum_true[CROSS_TOPIC]),
        n_unknown_topic=len(stratum_true[UNKNOWN_TOPIC]),
        delta_cross_topic=delta_cross,
        delta_same_topic=delta_same,
        v1_cross_topic_overall=v1_cross_topic_overall,
        v1_same_topic_overall=v1_same_topic_overall,
        timestamp=ts,
        per_pair=per_pair,
    )
    return result


# ──────────────────────────────── From AblationResult ─────────────────────────

def stratify_ablation_result(
    pairs: List[TextPair],
    ablation_result,              # AblationResult from evaluation/ablation.py
    v1_cross_topic_overall: Optional[float] = None,
    v1_same_topic_overall: Optional[float] = None,
) -> StratifiedResult:
    """Convenience wrapper: stratify a pre-computed AblationResult.

    Parameters
    ----------
    pairs            : The same TextPair list used to generate ``ablation_result``.
    ablation_result  : Output from any ``run_*`` function in ``ablation.py``.
    v1_cross_topic_overall / v1_same_topic_overall : V1 baselines for delta.

    Returns
    -------
    StratifiedResult
    """
    return evaluate_stratified(
        pairs=pairs,
        true_labels=ablation_result.true_labels,
        pred_scores=ablation_result.pred_scores,
        config_name=ablation_result.config_name,
        v1_cross_topic_overall=v1_cross_topic_overall,
        v1_same_topic_overall=v1_same_topic_overall,
    )


# ──────────────────────────────── Comparison ──────────────────────────────────

def compare_v2_vs_v1(
    v2_result: StratifiedResult,
    v1_result: StratifiedResult,
) -> Dict[str, Any]:
    """Build a comparison dict of V2 vs V1 across all strata.

    Parameters
    ----------
    v2_result : StratifiedResult for the FULL_V2 config.
    v1_result : StratifiedResult for the V1_BASELINE config.

    Returns
    -------
    Dict with per-stratum overall scores and deltas (V2 − V1).
    """
    strata = [
        ("all",           "all_metrics"),
        ("same_topic",    "same_topic_metrics"),
        ("cross_topic",   "cross_topic_metrics"),
        ("unknown_topic", "unknown_topic_metrics"),
    ]

    comparison: Dict[str, Any] = {
        "v2_config":  v2_result.config_name,
        "v1_config":  v1_result.config_name,
        "timestamp":  datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }

    for label, attr in strata:
        v2_m = getattr(v2_result, attr)
        v1_m = getattr(v1_result, attr)
        delta = round(v2_m["overall"] - v1_m["overall"], 3)
        comparison[label] = {
            "v2_overall":     v2_m["overall"],
            "v1_overall":     v1_m["overall"],
            "delta":          delta,
            "v2_all_metrics": v2_m,
            "v1_all_metrics": v1_m,
        }

    return comparison


# ──────────────────────────────── JSON Persistence ────────────────────────────

def save_stratified(
    result: StratifiedResult,
    output_dir: Optional[str] = None,
) -> str:
    """Serialise ``result`` to JSON in ``output_dir``.

    Parameters
    ----------
    result     : Completed stratified evaluation.
    output_dir : Directory to write into.  Defaults to ``experiments/results/``.

    Returns
    -------
    Path to the written file.
    """
    out_dir = Path(output_dir or "experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (
        f"topic_stratified_{result.config_name.lower()}"
        f"_{result.timestamp.replace(':', '-')}.json"
    )
    fpath = out_dir / fname
    fpath.write_text(
        json.dumps(asdict(result), indent=2, default=str),
        encoding="utf-8",
    )
    return str(fpath)


def save_comparison(
    comparison: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> str:
    """Save a V2-vs-V1 comparison dict to JSON.

    Parameters
    ----------
    comparison : Output of ``compare_v2_vs_v1()``.
    output_dir : Directory to write into.  Defaults to ``experiments/results/``.

    Returns
    -------
    Path to the written file.
    """
    out_dir = Path(output_dir or "experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = comparison.get("timestamp", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    fname = f"v2_vs_v1_comparison_{ts.replace(':', '-')}.json"
    fpath = out_dir / fname
    fpath.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    return str(fpath)


# ──────────────────────────────── Pretty Printer ─────────────────────────────

def print_stratified_report(result: StratifiedResult) -> None:
    """Print a human-readable stratified evaluation report."""
    w = 60
    print("\n" + "═" * w)
    print(f" TOPIC-STRATIFIED EVALUATION: {result.config_name}")
    print("═" * w)
    print(f"  Total pairs   : {result.n_all}")
    print(f"  Same-topic    : {result.n_same_topic}")
    print(f"  Cross-topic   : {result.n_cross_topic}")
    print(f"  Unknown-topic : {result.n_unknown_topic}")
    print("─" * w)

    strata_info = [
        ("ALL PAIRS",           result.all_metrics,          result.n_all),
        ("SAME-TOPIC",          result.same_topic_metrics,   result.n_same_topic),
        ("CROSS-TOPIC",         result.cross_topic_metrics,  result.n_cross_topic),
        ("UNKNOWN-TOPIC",       result.unknown_topic_metrics, result.n_unknown_topic),
    ]

    for label, metrics, n in strata_info:
        if n == 0:
            print(f"\n  {label}: (no pairs in this stratum)")
            continue
        print(f"\n  {label}  (n={n})")
        for k, v in metrics.items():
            print(f"    {k:<12}: {v:.3f}")

    print("─" * w)
    if result.delta_cross_topic is not None:
        sign = "+" if result.delta_cross_topic >= 0 else ""
        v1_note = (
            f" (V1 cross-topic overall: {result.v1_cross_topic_overall:.3f})"
            if result.v1_cross_topic_overall is not None
            else ""
        )
        print(
            f"  V2 vs V1 Δ cross-topic overall: "
            f"{sign}{result.delta_cross_topic:.3f}{v1_note}"
        )
    if result.delta_same_topic is not None:
        sign = "+" if result.delta_same_topic >= 0 else ""
        v1_note = (
            f" (V1 same-topic overall: {result.v1_same_topic_overall:.3f})"
            if result.v1_same_topic_overall is not None
            else ""
        )
        print(
            f"  V2 vs V1 Δ same-topic overall:  "
            f"{sign}{result.delta_same_topic:.3f}{v1_note}"
        )
    print("═" * w + "\n")
