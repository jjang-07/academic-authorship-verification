"""Ablation study runner.

Five configurations to isolate each V2 component's contribution:

╔══════════════════╦══════════╦═══════════╦═════════════╦════════════════════╗
║ Config           ║ Features ║ Retrieval ║ Multi-Agent ║ Expected           ║
╠══════════════════╬══════════╬═══════════╬═════════════╬════════════════════╣
║ V1_BASELINE      ║    ✓     ║     ✗     ║  ✗ (GB)     ║ ~0.707             ║
║ LIP_STYLE        ║    ✓     ║     ✗     ║  ✗ (1 LLM)  ║ Comparable to LIP  ║
║ NAIVE_MAD        ║    ✗     ║     ✗     ║  ✓ (debate) ║ Tests debate alone ║
║ FULL_V2          ║    ✓     ║     ✓     ║  ✓ (debate) ║ Best expected      ║
║ FULL_V2_TEXTONLY ║    ✗     ║     ✗     ║  ✓ (text)  ║ Close-reading only ║
╚══════════════════╩══════════╩═══════════╩═════════════╩════════════════════╝

Each runner
-----------
- Accepts ``List[TextPair]`` as input.
- Returns an ``AblationResult`` with metrics + per-pair metadata.
- Saves JSON to ``experiments/results/`` (configurable via ``output_dir``).

V1 Baseline approach
--------------------
Pair-level features are the absolute differences |f_a − f_b| across the
common numeric keys of both documents' ``extract_all()`` outputs.  A
GradientBoostingClassifier is trained with 5-fold cross-validation; the
out-of-fold (OOF) probability predictions are collected for evaluation — the
same protocol V1 used.

Naive MAD
---------
The full three-agent debate runs but the evidence packet is replaced with a
one-line note ("No stylometric analysis provided — reason from raw text only.")
so agents have no feature data to reference.  This isolates the value of the
structured evidence packet.

Cost control
------------
Pass ``max_pairs`` to cap the number of pairs evaluated (useful during
development when API costs matter).  Set to ``None`` for full evaluation.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from data.preprocessor import TextPair
from evaluation.metrics import evaluate_all, evaluate_extended, verdict_to_pred_score

# Lazy imports — only required for the relevant config
_AGENTS_IMPORTED = False


# ──────────────────────────────── Result Dataclass ────────────────────────────

@dataclass
class AblationResult:
    """Complete output from a single ablation configuration run.

    Fields
    ------
    config_name      : One of "V1_BASELINE", "LIP_STYLE", "NAIVE_MAD", "FULL_V2".
    metrics          : Dict with keys auc, c@1, f0.5_u, F1, brier, overall.
    pred_scores      : Same-author probability for each pair (in list order).
    true_labels      : Ground-truth labels (1 = same author, 0 = different).
    n_pairs          : Number of pairs evaluated.
    elapsed_seconds  : Wall-clock time for the full run.
    timestamp        : ISO-8601 UTC timestamp when the run completed.
    per_pair         : Per-pair metadata for debugging and stratified analysis.
                       Each entry is a dict with at least:
                       ``pair_id``, ``true_label``, ``pred_score``,
                       ``topic_a``, ``topic_b``.
    """
    config_name: str
    metrics: Dict[str, float]
    pred_scores: List[float]
    true_labels: List[int]
    n_pairs: int
    elapsed_seconds: float
    timestamp: str
    per_pair: List[Dict[str, Any]] = field(default_factory=list)


# ──────────────────────────────── JSON Persistence ────────────────────────────

def save_result(
    result: AblationResult,
    output_dir: Optional[str] = None,
) -> str:
    """Serialise ``result`` to JSON and write to ``output_dir``.

    Parameters
    ----------
    result     : The completed ablation run.
    output_dir : Directory to write into.  Defaults to ``experiments/results/``.

    Returns
    -------
    Path to the written file.
    """
    out_dir = Path(output_dir or "experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{result.config_name.lower()}_{result.timestamp.replace(':', '-')}.json"
    fpath = out_dir / fname
    fpath.write_text(
        json.dumps(asdict(result), indent=2, default=str),
        encoding="utf-8",
    )
    return str(fpath)


# ──────────────────────────────── Pair Helpers ────────────────────────────────

def _pair_meta(pair: TextPair, pred_score: float, pair_id: Optional[int] = None) -> Dict[str, Any]:
    """Build a per-pair metadata dict for the ablation result.

    pair_id: Synthetic index (0-based) when TextPair has no pair_id field.
               Pass from enumerate when calling.
    """
    return {
        "pair_id":    pair_id if pair_id is not None else "",
        "true_label": int(pair.label),
        "pred_score": round(pred_score, 4),
        "topic_a":    pair.topic_a or "",
        "topic_b":    pair.topic_b or "",
        "author_id":  pair.author_id or "",
        "dataset":    pair.source_dataset or "",
    }


def _clip_pairs(pairs: List[TextPair], max_pairs: Optional[int]) -> List[TextPair]:
    """Return first max_pairs from list. Caller should shuffle before passing if balance matters."""
    if max_pairs is not None and max_pairs < len(pairs):
        return pairs[:max_pairs]
    return pairs


# ──────────────────────────────── Mock responses for dry-run ──────────────────

def _mock_debate_result(pair: TextPair, pair_index: int) -> "DebateResult":
    """Build a mock DebateResult for dry-run mode (no API calls).

    Uses deterministic verdicts: ~50% correct (alternate by index) for non-trivial metrics.
    """
    from agents.base_agent import AgentResponse
    from agents.judge_agent import JudgeResponse
    from agents.skeptic_agent import SkepticResponse
    from debate.orchestrator import DebateResult

    gt = "SAME_AUTHOR" if pair.label == 1 else "DIFFERENT_AUTHOR"
    # Alternate: even index → match gt, odd → flip (yields ~50% accuracy)
    match_gt = (pair_index % 2) == 0
    verdict = gt if match_gt else ("DIFFERENT_AUTHOR" if gt == "SAME_AUTHOR" else "SAME_AUTHOR")
    conf = 0.75

    analyst = AgentResponse(
        verdict=verdict,
        confidence=conf,
        reasoning="[DRY-RUN] Mock analyst reasoning.",
        key_features_cited=["mock_feature"],
        raw_response="[DRY-RUN]",
        parse_ok=True,
    )
    skeptic = SkepticResponse(
        stance="AGREE" if match_gt else "PARTIALLY_DISAGREE",
        confidence=conf,
        challenges=["[DRY-RUN] Mock challenge."] if not match_gt else [],
        overlooked_evidence=[],
        revised_reasoning="[DRY-RUN] Mock skeptic reasoning.",
        raw_response="[DRY-RUN]",
        parse_ok=True,
    )
    judge = JudgeResponse(
        verdict=verdict,
        confidence=conf,
        decisive_factors=["[DRY-RUN] Mock decisive factor."],
        educator_summary="[DRY-RUN] Mock educator summary for pipeline verification.",
        agent_agreement="FULL" if match_gt else "PARTIAL",
        raw_response="[DRY-RUN]",
        parse_ok=True,
    )
    return DebateResult(
        analyst=analyst,
        skeptic=skeptic,
        judge=judge,
        rebuttal=None,
        final_verdict=verdict,
        final_confidence=conf,
        elapsed_seconds=0.01,
    )


def _mock_agent_response(pair: TextPair, pair_index: int) -> "AgentResponse":
    """Build a mock AgentResponse for LIP-style dry-run."""
    from agents.base_agent import AgentResponse

    gt = "SAME_AUTHOR" if pair.label == 1 else "DIFFERENT_AUTHOR"
    match_gt = (pair_index % 2) == 0
    verdict = gt if match_gt else ("DIFFERENT_AUTHOR" if gt == "SAME_AUTHOR" else "SAME_AUTHOR")
    return AgentResponse(
        verdict=verdict,
        confidence=0.75,
        reasoning="[DRY-RUN] Mock analyst reasoning.",
        key_features_cited=["mock_feature"],
        raw_response="[DRY-RUN]",
        parse_ok=True,
    )


# ──────────────────────────────── V1 Baseline ────────────────────────────────

def _build_pair_feature_matrix(
    pairs: List[TextPair],
    cefr_dict: Optional[Dict] = None,
) -> Tuple[np.ndarray, List[int], List[str]]:
    """Extract pair-level features as |f_a − f_b| deltas.

    Returns
    -------
    X           : (n_pairs × n_features) float array
    true_labels : list of int
    feat_names  : ordered list of feature key names
    """
    from features.handcrafted import extract_all

    all_feat_dicts: List[Tuple[Dict, Dict]] = []
    for pair in pairs:
        fa = extract_all(pair.text_a, cefr_dict=cefr_dict)
        fb = extract_all(pair.text_b, cefr_dict=cefr_dict)
        all_feat_dicts.append((fa, fb))

    # Gather common numeric keys (intersection)
    key_sets = [
        {k for k, v in fa.items() if isinstance(v, (int, float))}
        & {k for k, v in fb.items() if isinstance(v, (int, float))}
        for fa, fb in all_feat_dicts
    ]
    common_keys = sorted(set.intersection(*key_sets)) if key_sets else []

    X_rows: List[List[float]] = []
    true_labels: List[int] = []
    for (fa, fb), pair in zip(all_feat_dicts, pairs):
        row = [abs(float(fa.get(k, 0.0)) - float(fb.get(k, 0.0))) for k in common_keys]
        X_rows.append(row)
        true_labels.append(int(pair.label))

    X = np.array(X_rows, dtype=float)
    # Replace NaN/inf with 0
    X = np.where(np.isfinite(X), X, 0.0)
    return X, true_labels, common_keys


def run_v1_baseline(
    pairs: List[TextPair],
    cefr_dict: Optional[Dict] = None,
    n_folds: int = 5,
    seed: int = 42,
    max_pairs: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> AblationResult:
    """V1 Baseline: GradientBoosting on pair-level feature vectors only.

    Uses out-of-fold predictions from ``n_folds``-fold cross-validation so
    every pair gets a held-out probability score.  This exactly mirrors the
    V1 evaluation protocol (no data leakage).

    Parameters
    ----------
    pairs      : Text pairs to evaluate.
    cefr_dict  : CEFR wordlist (optional; skips CEFR features if None).
    n_folds    : Number of CV folds.  Default 5 (matches V1).
    seed       : Random seed for reproducibility.
    max_pairs  : Cap on pairs processed (for development cost control).
    output_dir : Where to save JSON results.

    Returns
    -------
    AblationResult
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold

    pairs = _clip_pairs(pairs, max_pairs)
    t0 = time.time()

    print(f"[V1_BASELINE] Extracting features for {len(pairs)} pairs …")
    X, true_labels, feat_names = _build_pair_feature_matrix(pairs, cefr_dict)
    print(f"[V1_BASELINE] Feature matrix: {X.shape}  ({len(feat_names)} features)")

    y = np.array(true_labels)
    oof_scores = np.full(len(pairs), 0.5)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        clf = GradientBoostingClassifier(random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X[train_idx], y[train_idx])
        oof_scores[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
        print(f"[V1_BASELINE] Fold {fold}/{n_folds} complete")

    pred_scores = oof_scores.tolist()
    metrics = evaluate_all(true_labels, pred_scores)
    elapsed = time.time() - t0
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    per_pair = [_pair_meta(p, s, i) for i, (p, s) in enumerate(zip(pairs, pred_scores))]

    result = AblationResult(
        config_name="V1_BASELINE",
        metrics=metrics,
        pred_scores=pred_scores,
        true_labels=true_labels,
        n_pairs=len(pairs),
        elapsed_seconds=round(elapsed, 2),
        timestamp=ts,
        per_pair=per_pair,
    )
    fpath = save_result(result, output_dir)
    print(f"[V1_BASELINE] Saved → {fpath}")
    print(f"[V1_BASELINE] Metrics: {metrics}")
    return result


# ──────────────────────────────── LIP-Style (single agent) ───────────────────

def run_lip_style(
    pairs: List[TextPair],
    cefr_dict: Optional[Dict] = None,
    max_pairs: Optional[int] = None,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
    delay_between_pairs: float = 0.0,
) -> AblationResult:
    """LIP-Style: single Analyst agent + evidence packet, no debate.

    Replicates the Huang et al. (2024) LIP setup:
    structured stylometric features → single LLM → verdict.
    No adversarial debate; no retrieval.

    Parameters
    ----------
    pairs      : Text pairs to evaluate.
    cefr_dict  : CEFR wordlist (optional).
    max_pairs  : Cap on pairs for cost control.
    output_dir : Where to save JSON results.
    dry_run    : If True, skip API calls and use mock responses.
    """
    from agents.analyst_agent import AnalystAgent
    from features.evidence_packet import format_evidence_packet
    from features.handcrafted import extract_all

    pairs = _clip_pairs(pairs, max_pairs)
    t0 = time.time()

    analyst = AnalystAgent() if not dry_run else None
    pred_scores: List[float] = []
    true_labels: List[int] = []
    per_pair: List[Dict[str, Any]] = []

    for i, pair in enumerate(pairs, 1):
        print(f"[LIP_STYLE] Pair {i}/{len(pairs)} …" + (" (dry-run)" if dry_run else ""))
        try:
            if dry_run:
                response = _mock_agent_response(pair, i - 1)
            else:
                fa = extract_all(pair.text_a, cefr_dict=cefr_dict)
                fb = extract_all(pair.text_b, cefr_dict=cefr_dict)
                packet = format_evidence_packet(fa, fb)
                response = analyst.analyze(pair, packet)
            score = verdict_to_pred_score(response.verdict, response.confidence)
            meta = _pair_meta(pair, score, pair_id=i - 1)
            meta["analyst_verdict"]     = response.verdict
            meta["analyst_confidence"]  = response.confidence
            meta["analyst_reasoning"]   = response.reasoning[:300] if response.reasoning else ""
        except Exception as exc:
            print(f"[LIP_STYLE] ERROR pair {i - 1} after all retries: {exc}", file=sys.stderr)
            score = 0.5
            meta = _pair_meta(pair, score, pair_id=i - 1)
            meta["error"] = str(exc)[:500]
        true_labels.append(int(pair.label))
        pred_scores.append(score)
        per_pair.append(meta)
        if not dry_run and delay_between_pairs > 0 and i < len(pairs):
            time.sleep(delay_between_pairs)

    metrics = evaluate_all(true_labels, pred_scores)
    elapsed = time.time() - t0
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    result = AblationResult(
        config_name="LIP_STYLE",
        metrics=metrics,
        pred_scores=pred_scores,
        true_labels=true_labels,
        n_pairs=len(pairs),
        elapsed_seconds=round(elapsed, 2),
        timestamp=ts,
        per_pair=per_pair,
    )
    fpath = save_result(result, output_dir)
    print(f"[LIP_STYLE] Saved → {fpath}")
    print(f"[LIP_STYLE] Metrics: {metrics}")
    return result


# ──────────────────────────────── Naive MAD (debate, no features) ────────────

_NAIVE_MAD_PACKET = (
    "STYLOMETRIC ANALYSIS\n"
    "====================\n"
    "[No stylometric analysis provided — agents must reason from raw text only.]\n"
    "\n"
    "NOTE: This is a NAIVE_MAD configuration. Feature vectors are intentionally\n"
    "omitted to test whether multi-agent debate alone adds value over a single\n"
    "LLM pass.\n"
)


def run_naive_mad(
    pairs: List[TextPair],
    max_rounds: int = 1,
    max_pairs: Optional[int] = None,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
    delay_between_pairs: float = 0.0,
) -> AblationResult:
    """Naive MAD: full three-agent debate WITHOUT stylometric features.

    Passes a placeholder evidence packet so agents can only reason from the
    raw text excerpts.  Isolates the contribution of the structured evidence
    packet by comparing this against FULL_V2.

    Parameters
    ----------
    pairs      : Text pairs to evaluate.
    max_rounds : Debate rounds (1 or 2).
    max_pairs  : Cap on pairs for cost control.
    output_dir : Where to save JSON results.
    dry_run    : If True, skip API calls and use mock DebateResult.
    """
    from agents.analyst_agent import AnalystAgent
    from agents.judge_agent import JudgeAgent
    from agents.skeptic_agent import SkepticAgent
    from debate.orchestrator import DebateOrchestrator

    pairs = _clip_pairs(pairs, max_pairs)
    t0 = time.time()

    analyst  = AnalystAgent()
    skeptic  = SkepticAgent()
    judge    = JudgeAgent()
    orch     = DebateOrchestrator(analyst, skeptic, judge, max_rounds=max_rounds)

    pred_scores: List[float] = []
    true_labels: List[int] = []
    per_pair: List[Dict[str, Any]] = []

    for i, pair in enumerate(pairs, 1):
        print(f"[NAIVE_MAD] Pair {i}/{len(pairs)} …" + (" (dry-run)" if dry_run else ""))
        try:
            if dry_run:
                debate = _mock_debate_result(pair, i - 1)
            else:
                debate = orch.run(pair, _NAIVE_MAD_PACKET)
            score = verdict_to_pred_score(debate.final_verdict, debate.final_confidence)
            meta = _pair_meta(pair, score, pair_id=i - 1)
            meta["analyst_verdict"]    = debate.analyst.verdict
            meta["analyst_confidence"] = debate.analyst.confidence
            meta["judge_verdict"]      = debate.final_verdict
            meta["judge_confidence"]   = debate.final_confidence
        except Exception as exc:
            print(f"[NAIVE_MAD] ERROR pair {i - 1} after all retries: {exc}", file=sys.stderr)
            score = 0.5
            meta = _pair_meta(pair, score, pair_id=i - 1)
            meta["error"] = str(exc)[:500]
        true_labels.append(int(pair.label))
        pred_scores.append(score)
        per_pair.append(meta)
        if not dry_run and delay_between_pairs > 0 and i < len(pairs):
            time.sleep(delay_between_pairs)

    metrics = evaluate_all(true_labels, pred_scores)
    elapsed = time.time() - t0
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    result = AblationResult(
        config_name="NAIVE_MAD",
        metrics=metrics,
        pred_scores=pred_scores,
        true_labels=true_labels,
        n_pairs=len(pairs),
        elapsed_seconds=round(elapsed, 2),
        timestamp=ts,
        per_pair=per_pair,
    )
    fpath = save_result(result, output_dir)
    print(f"[NAIVE_MAD] Saved → {fpath}")
    print(f"[NAIVE_MAD] Metrics: {metrics}")
    return result


# ──────────────────────────────── Full V2 ─────────────────────────────────────

def run_full_v2(
    pairs: List[TextPair],
    cefr_dict: Optional[Dict] = None,
    max_rounds: int = 1,
    max_pairs: Optional[int] = None,
    output_dir: Optional[str] = None,
    retriever=None,
    dry_run: bool = False,
    delay_between_pairs: float = 0.0,
) -> AblationResult:
    """Full V2: features + evidence packet + retrieval (optional) + 3-agent debate.

    Parameters
    ----------
    pairs      : Text pairs to evaluate.
    cefr_dict  : CEFR wordlist (optional; skips CEFR features if None).
    max_rounds : Debate rounds (1 or 2).
    max_pairs  : Cap on pairs for cost control.
    output_dir : Where to save JSON results.
    retriever  : Optional ``Retriever`` instance from ``retrieval/retriever.py``.
                 If provided, retrieved context is passed to the Analyst.
    dry_run    : If True, skip API calls and use mock DebateResult.
    """
    from agents.analyst_agent import AnalystAgent
    from agents.judge_agent import JudgeAgent
    from agents.skeptic_agent import SkepticAgent
    from debate.orchestrator import DebateOrchestrator
    from features.evidence_packet import format_evidence_packet
    from features.handcrafted import extract_all

    pairs = _clip_pairs(pairs, max_pairs)
    t0 = time.time()

    analyst  = AnalystAgent()
    skeptic  = SkepticAgent()
    judge    = JudgeAgent()
    orch     = DebateOrchestrator(analyst, skeptic, judge, max_rounds=max_rounds)

    pred_scores: List[float] = []
    true_labels: List[int] = []
    per_pair: List[Dict[str, Any]] = []

    for i, pair in enumerate(pairs, 1):
        print(f"[FULL_V2] Pair {i}/{len(pairs)} …" + (" (dry-run)" if dry_run else ""))
        retrieved_context: Optional[str] = None
        try:
            if dry_run:
                debate = _mock_debate_result(pair, i - 1)
            else:
                fa = extract_all(pair.text_a, cefr_dict=cefr_dict)
                fb = extract_all(pair.text_b, cefr_dict=cefr_dict)
                packet = format_evidence_packet(fa, fb)
                if retriever is not None:
                    try:
                        retrieved_context = retriever.format_context(
                            query_features=fa, k=3
                        )
                    except Exception:
                        retrieved_context = None
                debate = orch.run(pair, packet, retrieved_context=retrieved_context)
            score = verdict_to_pred_score(debate.final_verdict, debate.final_confidence)
            meta = _pair_meta(pair, score, pair_id=i - 1)
            meta["analyst_verdict"]    = debate.analyst.verdict
            meta["analyst_confidence"] = debate.analyst.confidence
            meta["judge_verdict"]      = debate.final_verdict
            meta["judge_confidence"]   = debate.final_confidence
            meta["has_retrieval"]      = False if dry_run else (retrieved_context is not None)
        except Exception as exc:
            print(f"[FULL_V2] ERROR pair {i - 1} after all retries: {exc}", file=sys.stderr)
            score = 0.5
            meta = _pair_meta(pair, score, pair_id=i - 1)
            meta["error"] = str(exc)[:500]
            meta["has_retrieval"] = False
        true_labels.append(int(pair.label))
        pred_scores.append(score)
        per_pair.append(meta)
        if not dry_run and delay_between_pairs > 0 and i < len(pairs):
            time.sleep(delay_between_pairs)

    metrics = evaluate_all(true_labels, pred_scores)
    elapsed = time.time() - t0
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    result = AblationResult(
        config_name="FULL_V2",
        metrics=metrics,
        pred_scores=pred_scores,
        true_labels=true_labels,
        n_pairs=len(pairs),
        elapsed_seconds=round(elapsed, 2),
        timestamp=ts,
        per_pair=per_pair,
    )
    fpath = save_result(result, output_dir)
    print(f"[FULL_V2] Saved → {fpath}")
    print(f"[FULL_V2] Metrics: {metrics}")
    return result


# ──────────────────────────────── Full V2 Text-Only (GPT-5.4 / Claude / Judge) ─

_GPT5_ANALYST_MODEL = "gpt-5.4"
_GPT5_SKEPTIC_MODEL = "claude-opus-4-6"
_GPT5_JUDGE_MODEL = "gpt-5.4"


def _build_full_v2_textonly_jsonl_record(
    pair_id: int,
    ground_truth: str,
    elapsed_seconds: float,
    *,
    error: Optional[str] = None,
    debate: Any = None,
    correct: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build one FULL_V2_TEXTONLY JSONL object (full strings, no truncation).

    Schema is shared by Reuters (``run_experiment``) and student essays
    (``run_student_experiment``) so ``experiments/inspect_results.py`` works on both.
    """
    es = round(elapsed_seconds, 2)

    def _rebuttal_block(rebuttal: Any) -> Dict[str, Any]:
        if rebuttal is None:
            return {
                "analyst_rebuttal_verdict": None,
                "analyst_rebuttal_confidence": None,
                "analyst_rebuttal_reasoning": None,
                "analyst_rebuttal_key_features_cited": None,
                "analyst_rebuttal_raw_response": None,
            }
        return {
            "analyst_rebuttal_verdict": rebuttal.verdict,
            "analyst_rebuttal_confidence": rebuttal.confidence,
            "analyst_rebuttal_reasoning": rebuttal.reasoning or "",
            "analyst_rebuttal_key_features_cited": rebuttal.key_features_cited or [],
            "analyst_rebuttal_raw_response": rebuttal.raw_response or "",
        }

    if error is not None:
        out: Dict[str, Any] = {
            "pair_id": pair_id,
            "ground_truth": ground_truth,
            "error": error,
            "analyst_verdict": None,
            "analyst_confidence": None,
            "analyst_reasoning": None,
            "analyst_key_features_cited": None,
            "analyst_raw_response": None,
            "skeptic_stance": None,
            "skeptic_confidence": None,
            "skeptic_challenges": None,
            "skeptic_overlooked_evidence": None,
            "skeptic_revised_reasoning": None,
            "skeptic_raw_response": None,
            "judge_verdict": None,
            "judge_confidence": None,
            "judge_decisive_factors": None,
            "judge_educator_summary": None,
            "judge_agent_agreement": None,
            "judge_raw_response": None,
            "final_verdict": None,
            "correct": None,
            "elapsed_seconds": es,
        }
        out.update(_rebuttal_block(None))
        return out

    a, s, j = debate.analyst, debate.skeptic, debate.judge
    out = {
        "pair_id": pair_id,
        "ground_truth": ground_truth,
        "analyst_verdict": a.verdict,
        "analyst_confidence": a.confidence,
        "analyst_reasoning": a.reasoning or "",
        "analyst_key_features_cited": a.key_features_cited or [],
        "analyst_raw_response": a.raw_response or "",
        "skeptic_stance": s.stance,
        "skeptic_confidence": s.confidence,
        "skeptic_challenges": s.challenges or [],
        "skeptic_overlooked_evidence": s.overlooked_evidence or [],
        "skeptic_revised_reasoning": s.revised_reasoning or "",
        "skeptic_raw_response": s.raw_response or "",
        "judge_verdict": j.verdict,
        "judge_confidence": j.confidence,
        "judge_decisive_factors": j.decisive_factors or [],
        "judge_educator_summary": j.educator_summary or "",
        "judge_agent_agreement": j.agent_agreement,
        "judge_raw_response": j.raw_response or "",
        "final_verdict": debate.final_verdict,
        "correct": correct,
        "elapsed_seconds": es,
    }
    out.update(_rebuttal_block(debate.rebuttal))
    return out


def run_full_v2_textonly(
    pairs: List[TextPair],
    max_rounds: int = 1,
    max_pairs: Optional[int] = None,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
    delay_between_pairs: float = 0.0,
    dataset_context: Optional[str] = None,
) -> AblationResult:
    """Full V2 Text-Only: close-reading pipeline with heterogeneous models.

    Analyst: gpt-5.4 (OpenAI Responses API, reasoning_effort=high)
    Skeptic: claude-opus-4-6 (Anthropic)
    Judge:   gpt-5.4 (OpenAI Responses API, reasoning_effort=high)

    No evidence packet — all three agents reason from raw text only.
    Saves a JSONL detail file per pair for post-hoc inspection (full reasoning,
    lists, educator summary, revised reasoning, rebuttal when rounds>1, and
    ``*_raw_response`` fields — no truncation).
    Uses evaluate_extended for same/different-author accuracy and abstention rate.

    Caller should shuffle pairs with a fixed seed before passing to ensure
    balanced same/different-author representation when using max_pairs.

    dry_run : If True, skip API calls and use mock DebateResult (JSONL still written).
    dataset_context : Optional tag (e.g. ``\"student_essays\"``) passed to
        ``DebateOrchestratorTextOnly`` for prompt injection.
    """
    from agents.analyst_agent_textonly import AnalystAgentTextOnly
    from agents.judge_agent_textonly import JudgeAgentTextOnly
    from agents.skeptic_agent_textonly import SkepticAgentTextOnly
    from debate.orchestrator_textonly import DebateOrchestratorTextOnly

    pairs = _clip_pairs(pairs, max_pairs)

    t0 = time.time()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir or "experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"full_v2_textonly_details_{ts}.jsonl"

    analyst = AnalystAgentTextOnly(model=_GPT5_ANALYST_MODEL)
    skeptic = SkepticAgentTextOnly(model=_GPT5_SKEPTIC_MODEL)
    judge = JudgeAgentTextOnly(model=_GPT5_JUDGE_MODEL)
    orch = DebateOrchestratorTextOnly(
        analyst,
        skeptic,
        judge,
        max_rounds=max_rounds,
        verbose=False,
        dataset_context=dataset_context,
    )

    pred_scores: List[float] = []
    true_labels: List[int] = []
    per_pair: List[Dict[str, Any]] = []

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for i, pair in enumerate(pairs, 1):
            print(f"[FULL_V2_TEXTONLY] Pair {i}/{len(pairs)} …" + (" (dry-run)" if dry_run else ""))
            pair_t0 = time.time()
            gt = "SAME_AUTHOR" if pair.label == 1 else "DIFFERENT_AUTHOR"

            try:
                if dry_run:
                    debate = _mock_debate_result(pair, i - 1)
                else:
                    debate = orch.run(pair)
            except Exception as exc:
                elapsed = time.time() - pair_t0
                err_msg = str(exc)[:500]
                print(f"[FULL_V2_TEXTONLY] ERROR pair {i - 1} after all retries: {err_msg}", file=sys.stderr)
                true_labels.append(int(pair.label))
                pred_scores.append(0.5)
                meta = _pair_meta(pair, 0.5, pair_id=i - 1)
                meta["error"] = err_msg
                meta["per_pair_elapsed"] = round(elapsed, 2)
                per_pair.append(meta)
                detail = _build_full_v2_textonly_jsonl_record(
                    i - 1,
                    gt,
                    elapsed,
                    error=err_msg,
                )
                jf.write(json.dumps(detail, default=str) + "\n")
                jf.flush()
                if not dry_run and delay_between_pairs > 0 and i < len(pairs):
                    time.sleep(delay_between_pairs)
                continue

            elapsed = time.time() - pair_t0
            score = verdict_to_pred_score(debate.final_verdict, debate.final_confidence)
            correct = debate.judge.verdict == gt

            true_labels.append(int(pair.label))
            pred_scores.append(score)

            meta = _pair_meta(pair, score, pair_id=i - 1)
            meta["analyst_verdict"] = debate.analyst.verdict
            meta["analyst_confidence"] = debate.analyst.confidence
            meta["judge_verdict"] = debate.final_verdict
            meta["judge_confidence"] = debate.final_confidence
            meta["per_pair_elapsed"] = round(elapsed, 2)
            per_pair.append(meta)

            detail = _build_full_v2_textonly_jsonl_record(
                i - 1,
                gt,
                elapsed,
                debate=debate,
                correct=correct,
            )
            jf.write(json.dumps(detail, default=str) + "\n")
            jf.flush()
            if not dry_run and delay_between_pairs > 0 and i < len(pairs):
                time.sleep(delay_between_pairs)

    metrics = evaluate_extended(true_labels, pred_scores)
    elapsed_total = time.time() - t0
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    result = AblationResult(
        config_name="FULL_V2_TEXTONLY",
        metrics=metrics,
        pred_scores=pred_scores,
        true_labels=true_labels,
        n_pairs=len(pairs),
        elapsed_seconds=round(elapsed_total, 2),
        timestamp=ts,
        per_pair=per_pair,
    )
    fpath = save_result(result, output_dir)
    print(f"[FULL_V2_TEXTONLY] Saved → {fpath}")
    print(f"[FULL_V2_TEXTONLY] Details saved → {jsonl_path}")
    print(f"[FULL_V2_TEXTONLY] Metrics: {metrics}")
    return result


# ──────────────────────────────── Convenience Runner ─────────────────────────

def run_all(
    pairs: List[TextPair],
    cefr_dict: Optional[Dict] = None,
    configs: Optional[List[str]] = None,
    max_pairs: Optional[int] = None,
    max_rounds: int = 1,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
    delay_between_pairs: float = 0.0,
) -> Dict[str, AblationResult]:
    """Run multiple ablation configurations and return a results dict.

    Parameters
    ----------
    pairs      : Text pairs to evaluate.
    cefr_dict  : CEFR wordlist (optional).
    configs    : Subset of configs to run, e.g. ``["V1_BASELINE", "FULL_V2"]``.
                 If None, all four are run.
    max_pairs  : Cap on pairs for cost control (applied to LLM configs only).
    max_rounds : Debate rounds for MAD-based configs.
    output_dir : Where to save JSON results.
    dry_run    : If True, LLM configs use mock responses (no API calls).

    Returns
    -------
    Dict mapping config_name → AblationResult.
    """
    all_configs = ["V1_BASELINE", "LIP_STYLE", "NAIVE_MAD", "FULL_V2", "FULL_V2_TEXTONLY"]
    selected = configs if configs is not None else all_configs
    unknown = set(selected) - set(all_configs)
    if unknown:
        raise ValueError(f"Unknown config(s): {unknown}.  Choose from {all_configs}.")

    results: Dict[str, AblationResult] = {}

    if "V1_BASELINE" in selected:
        # V1 baseline runs on all pairs (no API calls)
        results["V1_BASELINE"] = run_v1_baseline(
            pairs, cefr_dict=cefr_dict, output_dir=output_dir
        )

    if "LIP_STYLE" in selected:
        results["LIP_STYLE"] = run_lip_style(
            pairs, cefr_dict=cefr_dict, max_pairs=max_pairs,
            output_dir=output_dir, dry_run=dry_run,
            delay_between_pairs=delay_between_pairs
        )

    if "NAIVE_MAD" in selected:
        results["NAIVE_MAD"] = run_naive_mad(
            pairs, max_rounds=max_rounds, max_pairs=max_pairs,
            output_dir=output_dir, dry_run=dry_run,
            delay_between_pairs=delay_between_pairs
        )

    if "FULL_V2" in selected:
        results["FULL_V2"] = run_full_v2(
            pairs, cefr_dict=cefr_dict, max_rounds=max_rounds,
            max_pairs=max_pairs, output_dir=output_dir, dry_run=dry_run,
            delay_between_pairs=delay_between_pairs
        )

    if "FULL_V2_TEXTONLY" in selected:
        results["FULL_V2_TEXTONLY"] = run_full_v2_textonly(
            pairs, max_rounds=max_rounds,
            max_pairs=max_pairs, output_dir=output_dir, dry_run=dry_run,
            delay_between_pairs=delay_between_pairs
        )

    # Print comparison table
    _print_comparison(results)

    # Save combined summary
    summary = {
        name: {
            "metrics": r.metrics,
            "n_pairs": r.n_pairs,
            "elapsed_seconds": r.elapsed_seconds,
            "timestamp": r.timestamp,
        }
        for name, r in results.items()
    }
    out_dir = Path(output_dir or "experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = out_dir / f"ablation_summary_{ts}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[run_all] Summary saved → {summary_path}")

    return results


def _print_comparison(results: Dict[str, AblationResult]) -> None:
    """Print a human-readable comparison table of ablation metrics."""
    if not results:
        return
    header_keys = ["auc", "c@1", "f0.5_u", "F1", "brier", "overall"]
    col_w = 9
    name_w = 14
    print("\n" + "═" * (name_w + col_w * len(header_keys) + 2))
    print(f"{'CONFIG':<{name_w}}" + "".join(f"{k:>{col_w}}" for k in header_keys))
    print("─" * (name_w + col_w * len(header_keys) + 2))
    for name, result in results.items():
        row = f"{name:<{name_w}}"
        for k in header_keys:
            val = result.metrics.get(k, float("nan"))
            row += f"{val:>{col_w}.3f}"
        print(row)
    print("═" * (name_w + col_w * len(header_keys) + 2) + "\n")
