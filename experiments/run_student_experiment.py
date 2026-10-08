#!/usr/bin/env python3
"""Run FULL_V2_TEXTONLY on the student-essay corpus (all balanced pairs).

Uses the same ``run_full_v2_textonly`` pipeline as ``run_experiment.py`` (no
``max_pairs`` cap — every pair from ``load_student_essay_pairs`` is evaluated).

Results are written under ``experiments/results/student_essays/`` by default.

Agent system prompts use ``dataset_context='student_essays'`` so the wire-service
warning blocks are replaced with an **ACADEMIC ESSAY WARNING** (see
``debate.textonly_dataset_prompt``).

JSONL details use the same schema as Reuters ``FULL_V2_TEXTONLY`` runs
(``evaluation.ablation.run_full_v2_textonly``): full analyst reasoning,
skeptic challenges / overlooked / revised reasoning, judge decisive factors
and educator summary, optional round-2 rebuttal fields, and ``*_raw_response``
blobs with no truncation — so ``python experiments/inspect_results.py <jsonl>``
works the same as for Reuters experiment outputs.

Usage
-----
    python experiments/run_student_experiment.py
    python experiments/run_student_experiment.py --seed 123 --delay 5
    python experiments/run_student_experiment.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from data.student_essay_loader import load_student_essay_pairs
from debate.textonly_dataset_prompt import DATASET_CONTEXT_STUDENT_ESSAYS
from evaluation.ablation import run_full_v2_textonly


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FULL_V2_TEXTONLY on student essays (full corpus, no sampling).",
    )
    p.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Student essay root (default: data/raw/student_essays).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "experiments" / "results" / "student_essays",
        help="Where to save JSON + JSONL (default: experiments/results/student_essays).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="Shuffle seed for pair order (default: config.RANDOM_SEED).",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=1,
        choices=[1, 2],
        help="Debate rounds (1 or 2).",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0.0,
        metavar="SEC",
        help="Seconds to sleep between API pairs (default: 0).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls; write mock JSONL for pipeline check.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STUDENT ESSAYS — FULL_V2_TEXTONLY")
    print("=" * 60)
    print(f"  output_dir : {out_dir}")
    print(f"  seed       : {args.seed}")
    print(f"  rounds     : {args.rounds}")
    print(f"  delay      : {args.delay}s")
    print(f"  dry_run    : {args.dry_run}")
    print(f"  Dataset context: {DATASET_CONTEXT_STUDENT_ESSAYS}")
    print("=" * 60)

    t0 = time.time()
    try:
        pairs = load_student_essay_pairs(
            base_dir=args.base_dir,
            seed=args.seed,
            print_summary=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    print(f"\n[student] Running FULL_V2_TEXTONLY on all {len(shuffled)} pairs (no max_pairs cap)…\n")

    result = run_full_v2_textonly(
        shuffled,
        max_rounds=args.rounds,
        max_pairs=None,
        output_dir=str(out_dir),
        dry_run=args.dry_run,
        delay_between_pairs=args.delay,
        dataset_context=DATASET_CONTEXT_STUDENT_ESSAYS,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = out_dir / f"student_run_summary_{ts}.json"
    summary = {
        "experiment": "FULL_V2_TEXTONLY_STUDENT_ESSAYS",
        "timestamp": ts,
        "n_pairs": result.n_pairs,
        "metrics": result.metrics,
        "elapsed_seconds": result.elapsed_seconds,
        "seed": args.seed,
        "rounds": args.rounds,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[student] Run summary saved → {summary_path}")

    elapsed = round(time.time() - t0, 1)
    print(f"[student] Wall time (including load): {elapsed}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
