#!/usr/bin/env python3
"""Three-agent text-only debate with minimal (naive) system prompts — no ICL / stylometry blocks.

Uses the same 100 Reuters pairs as Run 2: ``load_reuters(..., seed=123)`` then
shuffle with seed 123, then first 100 pairs (same as ``run_experiment.py --seed 123``).

All three agents use the same OpenAI model (default ``gpt-5.4``; override with
``--model``, e.g. ``o4-mini`` for Analyst, Skeptic, and Judge).

Each role’s system prompt is the user-specified naive text plus a short **format
appendix** (VERDICT/STANCE/FINAL_VERDICT headers only) so existing parsers keep
working — no wire-service warnings, topic checks, or six-dimension guidance.

Usage
-----
    python experiments/run_naive_debate.py
    python experiments/run_naive_debate.py --dry-run --max-pairs 3
    python experiments/run_naive_debate.py --delay 5

Requires OPENAI_API_KEY in .env (Anthropic not used).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from agents.analyst_agent_textonly import AnalystAgentTextOnly
from agents.judge_agent_textonly import JudgeAgentTextOnly
from agents.skeptic_agent_textonly import SkepticAgentTextOnly
from data.preprocessor import TextPair, load_reuters
from debate.orchestrator_textonly import DebateOrchestratorTextOnly
from evaluation.ablation import (
    AblationResult,
    _build_full_v2_textonly_jsonl_record,
    _clip_pairs,
    _mock_debate_result,
    _pair_meta,
    save_result,
)
from evaluation.metrics import evaluate_extended, verdict_to_pred_score
from evaluation.topic_stratified import print_stratified_report, save_stratified, stratify_ablation_result

_DEFAULT_DEBATE_MODEL = "gpt-5.4"

_CONFIG_NAME = "NAIVE_DEBATE_TEXTONLY"

# User-specified naive instructions + minimal headers required by existing parsers.
# (Texts for the Analyst are in the user message as TEXT A / TEXT B; [text] is instructional.)
NAIVE_ANALYST_SYSTEM = (
    "Determine whether the following two texts were written by the same author. "
    "Text A: [text]. Text B: [text]. "
    "Output SAME_AUTHOR or DIFFERENT_AUTHOR with a confidence score between 0.5 and 1.0 "
    "and brief reasoning.\n\n"
    "Use these exact section headings in your reply:\n"
    "VERDICT: SAME_AUTHOR or DIFFERENT_AUTHOR\n"
    "CONFIDENCE: <number between 0.5 and 1.0>\n"
    "KEY FEATURES: (short bullet list)\n"
    "REASONING: (brief)\n"
)

NAIVE_SKEPTIC_SYSTEM = (
    "Two texts were compared for authorship. Read the texts and the initial verdict below. "
    "Do you agree? Output AGREE, PARTIALLY_DISAGREE, or DISAGREE with brief reasoning.\n\n"
    "Use these exact section headings in your reply:\n"
    "STANCE: AGREE, PARTIALLY_DISAGREE, or DISAGREE\n"
    "CONFIDENCE: <number between 0.5 and 1.0>\n"
    "CHALLENGES: (bullet list)\n"
    "OVERLOOKED_EVIDENCE: (bullet list)\n"
    "REVISED_REASONING: (paragraph)\n"
)

NAIVE_JUDGE_SYSTEM = (
    "Read both texts and the debate below. Make a final determination: were these written "
    "by the same author? Output SAME_AUTHOR or DIFFERENT_AUTHOR with a confidence score "
    "between 0.5 and 1.0.\n\n"
    "Use these exact section headings in your reply:\n"
    "FINAL_VERDICT: SAME_AUTHOR or DIFFERENT_AUTHOR\n"
    "FINAL_CONFIDENCE: <number between 0.5 and 1.0>\n"
    "DECISIVE_FACTORS: (bullet list)\n"
    "EDUCATOR_SUMMARY: (short paragraph)\n"
    "AGENT_AGREEMENT: FULL, PARTIAL, or NONE\n"
)


def _shuffle_pairs(pairs: List[TextPair], seed: int) -> List[TextPair]:
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def run_naive_debate(
    *,
    seed: int,
    shuffle_seed: int,
    max_pairs: int,
    max_rounds: int,
    output_dir: Path,
    delay_between_pairs: float,
    dry_run: bool,
    model: str,
) -> AblationResult:
    print(f"\n[naive_debate] Loading Reuters from {config.REUTERS_DIR} (pairgen seed={seed}) …")
    pairs = load_reuters(config.REUTERS_DIR, seed=seed)
    shuffled = _shuffle_pairs(pairs, shuffle_seed)
    selected = _clip_pairs(shuffled, max_pairs)
    print(
        f"[naive_debate] Running {len(selected)} pairs "
        f"(shuffle_seed={shuffle_seed}; same order as run_experiment --seed {shuffle_seed})."
    )

    t0 = time.time()
    ts_jsonl = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"naive_debate_details_{ts_jsonl}.jsonl"

    analyst = AnalystAgentTextOnly(
        model=model,
        system_prompt=NAIVE_ANALYST_SYSTEM,
    )
    skeptic = SkepticAgentTextOnly(
        model=model,
        system_prompt=NAIVE_SKEPTIC_SYSTEM,
    )
    judge = JudgeAgentTextOnly(
        model=model,
        system_prompt=NAIVE_JUDGE_SYSTEM,
    )
    orch = DebateOrchestratorTextOnly(
        analyst,
        skeptic,
        judge,
        max_rounds=max_rounds,
        verbose=False,
        dataset_context=None,
    )

    pred_scores: List[float] = []
    true_labels: List[int] = []
    per_pair: List[Dict[str, Any]] = []

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for i, pair in enumerate(selected, 1):
            tag = " (dry-run)" if dry_run else ""
            print(f"[naive_debate] Pair {i}/{len(selected)} …{tag}")
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
                print(f"[naive_debate] ERROR pair {i - 1} after all retries: {err_msg}", file=sys.stderr)
                true_labels.append(int(pair.label))
                pred_scores.append(0.5)
                meta = _pair_meta(pair, 0.5, pair_id=i - 1)
                meta["error"] = err_msg
                meta["per_pair_elapsed"] = round(elapsed, 2)
                per_pair.append(meta)
                detail = _build_full_v2_textonly_jsonl_record(
                    i - 1, gt, elapsed, error=err_msg
                )
                jf.write(json.dumps(detail, default=str) + "\n")
                jf.flush()
                if not dry_run and delay_between_pairs > 0 and i < len(selected):
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
                i - 1, gt, elapsed, debate=debate, correct=correct
            )
            jf.write(json.dumps(detail, default=str) + "\n")
            jf.flush()

            if not dry_run and delay_between_pairs > 0 and i < len(selected):
                time.sleep(delay_between_pairs)

    metrics = evaluate_extended(true_labels, pred_scores)
    elapsed_total = time.time() - t0
    ts_done = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    result = AblationResult(
        config_name=_CONFIG_NAME,
        metrics=metrics,
        pred_scores=pred_scores,
        true_labels=true_labels,
        n_pairs=len(selected),
        elapsed_seconds=round(elapsed_total, 2),
        timestamp=ts_done,
        per_pair=per_pair,
    )
    summary_path = save_result(result, str(output_dir))
    print(f"\n[naive_debate] Details JSONL → {jsonl_path}")
    print(f"[naive_debate] Summary JSON   → {summary_path}")
    print(f"[naive_debate] Metrics: {metrics}")

    strat = stratify_ablation_result(selected, result)
    print_stratified_report(strat)
    strat_path = save_stratified(strat, output_dir=str(output_dir))
    print(f"[naive_debate] Topic-stratified JSON → {strat_path}")

    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Naive three-agent text-only debate (minimal system prompts).")
    p.add_argument("--seed", type=int, default=123, help="Reuters load_reuters seed (default 123 = Run 2).")
    p.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        metavar="N",
        help="Shuffle seed before clipping (default: same as --seed).",
    )
    p.add_argument("--max-pairs", type=int, default=100, help="Pairs after shuffle (default 100).")
    p.add_argument("--rounds", type=int, default=1, choices=[1, 2], help="Debate rounds (default 1).")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(config.EXPERIMENTS_DIR),
        help="Output directory (default config.EXPERIMENTS_DIR).",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=5.0,
        metavar="SEC",
        help="Sleep between pairs (default 5).",
    )
    p.add_argument("--dry-run", action="store_true", help="Skip APIs; write mock debate rows.")
    p.add_argument(
        "--model",
        type=str,
        default=_DEFAULT_DEBATE_MODEL,
        metavar="NAME",
        help=(
            "OpenAI model for Analyst, Skeptic, and Judge (default: gpt-5.4). "
            "Example: o4-mini uses Chat Completions (no reasoning_effort on this stack)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    shuffle_seed = args.shuffle_seed if args.shuffle_seed is not None else args.seed
    out = args.output_dir.resolve()
    print("=" * 60)
    print("NAIVE DEBATE — TEXT-ONLY (minimal system prompts)")
    print("=" * 60)
    print(f"  seed:         {args.seed}")
    print(f"  shuffle_seed: {shuffle_seed}")
    print(f"  max_pairs:    {args.max_pairs}")
    print(f"  rounds:       {args.rounds}")
    print(f"  output_dir:   {out}")
    print(f"  delay:        {args.delay}s")
    print(f"  dry_run:      {args.dry_run}")
    print(f"  model:        {args.model} (Analyst, Skeptic, Judge)")
    print("=" * 60)

    run_naive_debate(
        seed=args.seed,
        shuffle_seed=shuffle_seed,
        max_pairs=args.max_pairs,
        max_rounds=args.rounds,
        output_dir=out,
        delay_between_pairs=args.delay,
        dry_run=args.dry_run,
        model=args.model.strip(),
    )


if __name__ == "__main__":
    main()
