"""Main experiment runner — produces the paper's primary results table.

Execution order
---------------
1. Load Reuters-50-50 corpus (all available pairs).
2. Load CEFR wordlist (optional — skipped gracefully if absent).
3. V1_BASELINE  — GradientBoosting on |f_a − f_b| vectors, full corpus,
                  5-fold CV.  Establishes the benchmark score (~0.707).
4. LIP_STYLE    — Analyst agent only + evidence packet, no debate.
                  Cap: ``--max-pairs`` (default 100).
5. NAIVE_MAD    — Full three-agent debate, no features (blank packet).
                  Cap: ``--max-pairs``.
6. FULL_V2          — Full pipeline: features + evidence packet + debate.
                      Cap: ``--max-pairs``.
7. FULL_V2_TEXTONLY — Text-only close-reading (gpt-5.4 Analyst & Judge / claude-opus Skeptic).
                      Cap: ``--max-pairs``. Saves JSONL details per pair.
8. Topic-stratified breakdown for every config that produced predictions.
9. Final comparison table printed to stdout and saved as JSON.

Pairs are shuffled with the experiment ``--seed`` (default 42) before selecting
``max_pairs`` for LLM configs, ensuring balanced same-author and different-author
representation. This seed also drives ``load_reuters`` pair generation.

All per-config JSON results are saved to ``experiments/results/`` as they
finish, so a partial run is never lost.

Usage
-----
    # Full run (100 LLM pairs per config):
    python experiments/run_experiment.py

    # Quick smoke test (5 pairs):
    python experiments/run_experiment.py --max-pairs 5

    # Skip LLM configs, baseline only:
    python experiments/run_experiment.py --configs V1_BASELINE

    # Choose which configs to run (space-separated):
    python experiments/run_experiment.py --configs V1_BASELINE LIP_STYLE FULL_V2

    # Use two debate rounds:
    python experiments/run_experiment.py --rounds 2

    # Override output directory:
    python experiments/run_experiment.py --output-dir /tmp/av_results

    # Dry-run: skip API calls, use mock verdicts (verifies pipeline, JSONL, metrics):
    python experiments/run_experiment.py --dry-run --max-pairs 5

    # Reproducible pair order (Reuters load + shuffle; JSONL pair_id alignment):
    python experiments/run_experiment.py --seed 42

Environment
-----------
Requires OPENAI_API_KEY (and/or ANTHROPIC_API_KEY) in .env for any LLM config.
Set REUTERS_DIR and CEFR_WORDLIST_PATH in .env if they differ from the defaults
in config.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# ── ensure project root is on sys.path when run as a script ──────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from data.preprocessor import TextPair, load_reuters
from evaluation.ablation import (
    AblationResult,
    run_full_v2,
    run_full_v2_textonly,
    run_lip_style,
    run_naive_mad,
    run_v1_baseline,
)
from evaluation.topic_stratified import (
    print_stratified_report,
    save_stratified,
    stratify_ablation_result,
)

# V1 known cross/same topic baseline scores for delta computation
# (from V1 paper; update once you have stratified V1 numbers)
_V1_CROSS_TOPIC_OVERALL: Optional[float] = None   # set e.g. 0.641
_V1_SAME_TOPIC_OVERALL:  Optional[float] = None   # set e.g. 0.723

_ALL_CONFIGS = ["V1_BASELINE", "LIP_STYLE", "NAIVE_MAD", "FULL_V2", "FULL_V2_TEXTONLY"]
_METRIC_KEYS = ["auc", "c@1", "f0.5_u", "F1", "brier", "overall"]


# ──────────────────────────────── Loading ─────────────────────────────────────

def _load_pairs(reuters_dir: str, seed: int) -> List[TextPair]:
    print(f"\n[experiment] Loading Reuters-50-50 from {reuters_dir} …")
    pairs = load_reuters(reuters_dir, seed=seed)
    if not pairs:
        print(
            "[experiment] WARNING: load_reuters() returned 0 pairs.\n"
            f"  Check that REUTERS_DIR={reuters_dir} contains C50train and C50test.\n"
            "  Set REUTERS_DIR in .env if needed.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[experiment] Loaded {len(pairs)} pairs "
          f"({sum(p.label for p in pairs)} same-author, "
          f"{sum(1 - p.label for p in pairs)} different-author)")
    return pairs


def _shuffle_pairs(pairs: List[TextPair], seed: int) -> List[TextPair]:
    """Shuffle pairs with fixed seed for balanced same/different-author subset selection."""
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _load_cefr() -> Optional[Dict]:
    try:
        from data.load_cefr import get_cefr_dict
        cefr = get_cefr_dict()
        print(f"[experiment] CEFR wordlist loaded ({len(cefr)} entries).")
        return cefr
    except Exception as e:
        print(f"[experiment] CEFR wordlist unavailable ({e}). CEFR features will be skipped.")
        return None


# ──────────────────────────────── Comparison Table ───────────────────────────

_COL_W   = 9
_NAME_W  = 14
_RULE    = "═" * (_NAME_W + _COL_W * len(_METRIC_KEYS) + 2)
_DIVIDER = "─" * (_NAME_W + _COL_W * len(_METRIC_KEYS) + 2)


def _print_comparison(results: Dict[str, AblationResult]) -> None:
    """Print the main results table to stdout."""
    print("\n" + _RULE)
    print("MAIN RESULTS TABLE — V2 Authorship Verification Experiment")
    print(_RULE)
    header = f"{'CONFIG':<{_NAME_W}}" + "".join(f"{k:>{_COL_W}}" for k in _METRIC_KEYS)
    print(header)
    print(_DIVIDER)

    # Reference line for V1 expected score
    ref_line = f"{'V1 (reported)':<{_NAME_W}}" + "".join(
        f"{'~0.707':>{_COL_W}}" if k == "overall" else f"{'':>{_COL_W}}"
        for k in _METRIC_KEYS
    )
    print(ref_line)
    print(_DIVIDER)

    for name in _ALL_CONFIGS:
        if name not in results:
            continue
        r = results[name]
        row = f"{name:<{_NAME_W}}"
        for k in _METRIC_KEYS:
            val = r.metrics.get(k, float("nan"))
            row += f"{val:>{_COL_W}.3f}"
        n_tag = f"  (n={r.n_pairs})"
        print(row + n_tag)

    print(_RULE)

    # Highlight FULL_V2 vs V1_BASELINE delta
    if "FULL_V2" in results and "V1_BASELINE" in results:
        delta = round(
            results["FULL_V2"].metrics["overall"]
            - results["V1_BASELINE"].metrics["overall"],
            3,
        )
        sign = "+" if delta >= 0 else ""
        print(f"\n  FULL_V2 vs V1_BASELINE overall delta: {sign}{delta}")
    print()


def _save_summary(
    results: Dict[str, AblationResult],
    output_dir: str,
    ts: str,
) -> str:
    """Save a compact JSON summary of all configs to output_dir."""
    summary = {
        "experiment_timestamp": ts,
        "configs": {
            name: {
                "metrics":          r.metrics,
                "n_pairs":          r.n_pairs,
                "elapsed_seconds":  r.elapsed_seconds,
            }
            for name, r in results.items()
        },
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f"main_results_{ts}.json"
    fpath.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[experiment] Summary saved → {fpath}")
    return str(fpath)


# ──────────────────────────────── Main ────────────────────────────────────────

def run(
    configs:    List[str],
    max_pairs:  Optional[int],
    max_rounds: int,
    output_dir: str,
    seed:       int,
    dry_run:    bool = False,
    delay:      float = 5.0,
) -> Dict[str, AblationResult]:

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results: Dict[str, AblationResult] = {}
    t_total = time.time()

    # ── Data loading ─────────────────────────────────────────────────────────
    pairs    = _load_pairs(config.REUTERS_DIR, seed)
    cefr_dict = _load_cefr()

    # Shuffle with fixed seed so LLM configs get balanced same/different-author
    # representation when using max_pairs (rather than first 100 sequentially)
    shuffled_pairs = _shuffle_pairs(pairs, seed)

    # ── V1 Baseline (no API calls — run on full corpus) ───────────────────────
    if "V1_BASELINE" in configs:
        print("\n" + "─" * 60)
        print("RUNNING: V1_BASELINE (GradientBoosting, full corpus, 5-fold CV)")
        print("─" * 60)
        results["V1_BASELINE"] = run_v1_baseline(
            pairs,
            cefr_dict=cefr_dict,
            n_folds=5,
            seed=seed,
            output_dir=output_dir,
        )

    # ── LIP-Style (Analyst only, no debate) ──────────────────────────────────
    if "LIP_STYLE" in configs:
        print("\n" + "─" * 60)
        print(f"RUNNING: LIP_STYLE (Analyst only, max_pairs={max_pairs})" + (" [DRY-RUN]" if dry_run else ""))
        print("─" * 60)
        results["LIP_STYLE"] = run_lip_style(
            shuffled_pairs,
            cefr_dict=cefr_dict,
            max_pairs=max_pairs,
            output_dir=output_dir,
            dry_run=dry_run,
            delay_between_pairs=delay,
        )

    # ── Naive MAD (debate, no features) ──────────────────────────────────────
    if "NAIVE_MAD" in configs:
        print("\n" + "─" * 60)
        print(f"RUNNING: NAIVE_MAD (3-agent debate, no features, max_pairs={max_pairs})" + (" [DRY-RUN]" if dry_run else ""))
        print("─" * 60)
        results["NAIVE_MAD"] = run_naive_mad(
            shuffled_pairs,
            max_rounds=max_rounds,
            max_pairs=max_pairs,
            output_dir=output_dir,
            dry_run=dry_run,
            delay_between_pairs=delay,
        )

    # ── Full V2 (features + packet + debate) ─────────────────────────────────
    if "FULL_V2" in configs:
        print("\n" + "─" * 60)
        print(f"RUNNING: FULL_V2 (full pipeline, max_pairs={max_pairs}, rounds={max_rounds})" + (" [DRY-RUN]" if dry_run else ""))
        print("─" * 60)
        results["FULL_V2"] = run_full_v2(
            shuffled_pairs,
            cefr_dict=cefr_dict,
            max_rounds=max_rounds,
            max_pairs=max_pairs,
            output_dir=output_dir,
            dry_run=dry_run,
            delay_between_pairs=delay,
        )

    # ── Full V2 Text-Only (gpt-5.4 ×2 / claude-opus) ─────────────────────────
    if "FULL_V2_TEXTONLY" in configs:
        print("\n" + "─" * 60)
        print(f"RUNNING: FULL_V2_TEXTONLY (text-only, max_pairs={max_pairs}, rounds={max_rounds})" + (" [DRY-RUN]" if dry_run else ""))
        print("─" * 60)
        results["FULL_V2_TEXTONLY"] = run_full_v2_textonly(
            shuffled_pairs,
            max_rounds=max_rounds,
            max_pairs=max_pairs,
            output_dir=output_dir,
            dry_run=dry_run,
            delay_between_pairs=delay,
        )

    # ── Main comparison table ─────────────────────────────────────────────────
    _print_comparison(results)

    # ── Topic-stratified breakdown ────────────────────────────────────────────
    v1_result_for_delta = results.get("V1_BASELINE")
    strat_v1 = None
    if v1_result_for_delta:
        strat_v1 = stratify_ablation_result(pairs, v1_result_for_delta)

    for name, result in results.items():
        # V1 uses full corpus; LLM configs use shuffled subset
        pairs_for_result = (
            pairs[:result.n_pairs]
            if name == "V1_BASELINE"
            else shuffled_pairs[:result.n_pairs]
        )
        strat = stratify_ablation_result(
            pairs_for_result,
            result,
            v1_cross_topic_overall=_V1_CROSS_TOPIC_OVERALL,
            v1_same_topic_overall=_V1_SAME_TOPIC_OVERALL,
        )
        print_stratified_report(strat)
        save_stratified(strat, output_dir=output_dir)

    # ── V2 vs V1 stratified comparison ───────────────────────────────────────
    if strat_v1 and "FULL_V2" in results:
        from evaluation.topic_stratified import compare_v2_vs_v1, save_comparison
        strat_v2 = stratify_ablation_result(
            shuffled_pairs[:results["FULL_V2"].n_pairs],
            results["FULL_V2"],
        )
        comparison = compare_v2_vs_v1(strat_v2, strat_v1)
        saved = save_comparison(comparison, output_dir=output_dir)
        print(f"[experiment] V2-vs-V1 comparison saved → {saved}")

    # ── Summary JSON ─────────────────────────────────────────────────────────
    _save_summary(results, output_dir, ts)

    elapsed_total = round(time.time() - t_total, 1)
    print(f"\n[experiment] All done. Total wall time: {elapsed_total}s")
    return results


# ──────────────────────────────── CLI ─────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run V2 authorship verification ablation experiment.\n"
            "Produces the main results table for the paper."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=_ALL_CONFIGS,
        choices=_ALL_CONFIGS,
        metavar="CONFIG",
        help=(
            "Which ablation configs to run. "
            f"Default: all ({', '.join(_ALL_CONFIGS)}). "
            "Example: --configs V1_BASELINE FULL_V2"
        ),
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=100,
        metavar="N",
        help=(
            "Maximum pairs for LLM-based configs (LIP_STYLE, NAIVE_MAD, FULL_V2, FULL_V2_TEXTONLY). "
            "V1_BASELINE always uses the full corpus. Default: 100."
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of debate rounds for MAD-based configs (1 or 2). Default: 1.",
    )
    parser.add_argument(
        "--output-dir",
        default=config.EXPERIMENTS_DIR,
        metavar="DIR",
        help=(
            f"Directory to save result JSON files. "
            f"Default: {config.EXPERIMENTS_DIR} (from config / .env)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help=(
            "Random seed for Reuters pair generation (load_reuters) and for shuffling "
            "pairs before LLM subset selection — controls JSONL pair_id order. "
            "Default: 42 (independent of config.RANDOM_SEED / .env unless you pass the same value)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip API calls for LLM configs; use mock verdicts instead. "
            "Verifies evaluation pipeline, JSONL logging, metrics, and file saving without spending API credits."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        metavar="SEC",
        help="Fixed sleep (seconds) between pairs to avoid bursting API calls. Default: 5.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print("=" * 60)
    print("V2 AUTHORSHIP VERIFICATION — MAIN EXPERIMENT")
    print("=" * 60)
    print(f"  Configs:    {args.configs}")
    print(f"  max_pairs:  {args.max_pairs}  (LLM configs only)")
    print(f"  rounds:     {args.rounds}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  seed:       {args.seed}")
    print(f"  dry_run:    {args.dry_run}")
    print(f"  delay:      {args.delay}s (between pairs)")
    print("=" * 60)

    run(
        configs=args.configs,
        max_pairs=args.max_pairs,
        max_rounds=args.rounds,
        output_dir=args.output_dir,
        seed=args.seed,
        dry_run=args.dry_run,
        delay=args.delay,
    )
