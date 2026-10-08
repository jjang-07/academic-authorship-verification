#!/usr/bin/env python3
"""Single-agent naive GPT-5.4 baseline — minimal prompt, no stylometry / debate.

Uses the same Reuters pair ordering as ``run_experiment.py`` with ``--seed 123``
and ``--max-pairs 100`` (shuffle seed 123, then first 100 pairs), matching
Run 2 text-only experiments for direct comparison.

Text A/B are truncated to the same character budget as the text-only analyst
(2500 chars each).

Usage
-----
    python experiments/run_naive_llm.py
    python experiments/run_naive_llm.py --dry-run --max-pairs 5
    python experiments/run_naive_llm.py --delay 5

Requires OPENAI_API_KEY in .env.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from agents.base_agent import BaseAgent
from data.preprocessor import TextPair, load_reuters
from evaluation.ablation import _clip_pairs
from evaluation.metrics import evaluate_extended, verdict_to_pred_score

# Match text-only analyst preview so inputs align with Run 2 FULL_V2_TEXTONLY
_TEXT_PREVIEW_CHARS = 2500
_NAIVE_MODEL_DEFAULT = "gpt-5.4"


def _truncate(text: str, max_chars: int = _TEXT_PREVIEW_CHARS) -> Tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _shuffle_pairs(pairs: List[TextPair], seed: int) -> List[TextPair]:
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def build_naive_user_message(pair: TextPair) -> Tuple[str, bool, bool]:
    """Return (user_message, truncated_a, truncated_b)."""
    ta, a_trunc = _truncate(pair.text_a)
    tb, b_trunc = _truncate(pair.text_b)
    msg = (
        "Determine whether the following two texts were written by the same author. "
        f"Text A: {ta} Text B: {tb} "
        "Answer SAME_AUTHOR or DIFFERENT_AUTHOR with a confidence between 0.5 and 1.0."
    )
    return msg, a_trunc, b_trunc


_VERDICT_RE = re.compile(r"\b(SAME_AUTHOR|DIFFERENT_AUTHOR)\b", re.IGNORECASE)
_CONFIDENCE_LABEL_RE = re.compile(
    r"CONFIDENCE\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE
)


def parse_naive_response(raw: str) -> Tuple[str, float, bool]:
    """Parse verdict and confidence from free-form model output."""
    vm = _VERDICT_RE.search(raw)
    verdict = vm.group(1).upper() if vm else "UNKNOWN"

    conf = 0.5
    cm = _CONFIDENCE_LABEL_RE.search(raw)
    if cm:
        conf = float(cm.group(1))
    else:
        candidates: List[float] = []
        for m in re.finditer(r"0\.\d+|1(?:\.0+)?", raw):
            try:
                v = float(m.group(0))
                if 0.5 <= v <= 1.0:
                    candidates.append(v)
            except ValueError:
                continue
        if candidates:
            conf = candidates[-1]

    conf = max(0.5, min(1.0, conf))
    parse_ok = verdict in ("SAME_AUTHOR", "DIFFERENT_AUTHOR")
    return verdict, conf, parse_ok


def _jsonl_record(
    pair_id: int,
    pair: TextPair,
    *,
    ground_truth: str,
    naive_verdict: str,
    naive_confidence: float,
    correct: Optional[bool],
    parse_ok: bool,
    raw_response: str,
    elapsed_seconds: float,
    error: Optional[str],
    model: str,
    seed: int,
    shuffle_seed: int,
    text_a_truncated: bool,
    text_b_truncated: bool,
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "pair_id": pair_id,
        "ground_truth": ground_truth,
        "naive_verdict": naive_verdict,
        "naive_confidence": round(float(naive_confidence), 4),
        "correct": correct,
        "parse_ok": parse_ok,
        "raw_response": raw_response,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "model": model,
        "reuters_pairgen_seed": seed,
        "shuffle_seed": shuffle_seed,
        "max_text_chars_per_side": _TEXT_PREVIEW_CHARS,
        "text_a_truncated": text_a_truncated,
        "text_b_truncated": text_b_truncated,
        "author_id": pair.author_id,
        "source_dataset": pair.source_dataset,
    }
    if error:
        rec["error"] = error
    return rec


def run(
    *,
    seed: int,
    shuffle_seed: int,
    max_pairs: int,
    output_dir: Path,
    delay_s: float,
    dry_run: bool,
    model: str,
) -> Path:
    print(f"\n[naive_llm] Loading Reuters from {config.REUTERS_DIR} (pairgen seed={seed}) …")
    pairs = load_reuters(config.REUTERS_DIR, seed=seed)
    shuffled = _shuffle_pairs(pairs, shuffle_seed)
    selected = _clip_pairs(shuffled, max_pairs)
    print(
        f"[naive_llm] Using {len(selected)} pairs "
        f"(shuffle_seed={shuffle_seed}, same as run_experiment --seed {shuffle_seed})."
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"naive_llm_details_{ts}.jsonl"

    # No role guidance beyond an empty system message — user content is only the task line + texts.
    agent = BaseAgent(model=model, system_prompt="", temperature=config.AGENT_TEMPERATURE)

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for i, pair in enumerate(selected, start=1):
            tag = " (dry-run)" if dry_run else ""
            print(f"[naive_llm] Pair {i}/{len(selected)} …{tag}")
            t0 = time.time()
            gt = "SAME_AUTHOR" if pair.label == 1 else "DIFFERENT_AUTHOR"
            user_msg, a_trunc, b_trunc = build_naive_user_message(pair)

            raw = ""
            err: Optional[str] = None
            verdict, conf, p_ok = "UNKNOWN", 0.5, False

            try:
                if dry_run:
                    raw = (
                        f"VERDICT: {'SAME_AUTHOR' if i % 2 == 0 else 'DIFFERENT_AUTHOR'}\n"
                        f"CONFIDENCE: {0.55 + (i % 10) * 0.04}"
                    )
                    verdict, conf, p_ok = parse_naive_response(raw)
                else:
                    raw = agent._raw_call(user_msg)
                    verdict, conf, p_ok = parse_naive_response(raw)
            except Exception as exc:
                err = str(exc)[:800]

            elapsed = time.time() - t0
            correct: Optional[bool] = None
            if err is None and verdict in ("SAME_AUTHOR", "DIFFERENT_AUTHOR"):
                correct = verdict == gt

            rec = _jsonl_record(
                i - 1,
                pair,
                ground_truth=gt,
                naive_verdict=verdict,
                naive_confidence=conf,
                correct=correct,
                parse_ok=p_ok and err is None,
                raw_response=raw if raw else "",
                elapsed_seconds=elapsed,
                error=err,
                model=model,
                seed=seed,
                shuffle_seed=shuffle_seed,
                text_a_truncated=a_trunc,
                text_b_truncated=b_trunc,
            )
            jf.write(json.dumps(rec, default=str) + "\n")
            jf.flush()

            if err:
                print(f"[naive_llm] ERROR pair {i - 1}: {err[:200]}", file=sys.stderr)

            if not dry_run and delay_s > 0 and i < len(selected):
                time.sleep(delay_s)

    print(f"\n[naive_llm] Wrote → {jsonl_path}")
    return jsonl_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Naive single-call GPT-5.4 baseline (minimal prompt).")
    p.add_argument("--seed", type=int, default=123, help="Reuters load_reuters seed (default 123, matches Run 2).")
    p.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        metavar="N",
        help="Shuffle seed before clipping (default: same as --seed).",
    )
    p.add_argument("--max-pairs", type=int, default=100, help="Number of pairs after shuffle (default 100).")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(config.EXPERIMENTS_DIR).resolve(),
        help="Directory for naive_llm_details_*.jsonl",
    )
    p.add_argument("--delay", type=float, default=0.0, help="Seconds to sleep between API calls.")
    p.add_argument("--dry-run", action="store_true", help="Skip API; write mock lines.")
    p.add_argument("--model", type=str, default=_NAIVE_MODEL_DEFAULT, help="OpenAI model (default gpt-5.4).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    shuffle_seed = args.shuffle_seed if args.shuffle_seed is not None else args.seed
    path = run(
        seed=args.seed,
        shuffle_seed=shuffle_seed,
        max_pairs=args.max_pairs,
        output_dir=args.output_dir.resolve(),
        delay_s=args.delay,
        dry_run=args.dry_run,
        model=args.model,
    )

    # Metrics from written file (handles dry-run / errors consistently)
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    valid = [r for r in rows if not r.get("error") and r.get("naive_verdict") in ("SAME_AUTHOR", "DIFFERENT_AUTHOR")]
    if valid:
        y = [1 if r["ground_truth"] == "SAME_AUTHOR" else 0 for r in valid]
        scores = [verdict_to_pred_score(r["naive_verdict"], float(r["naive_confidence"])) for r in valid]
        m = evaluate_extended(y, scores)
        acc = sum(1 for r in valid if r.get("correct")) / len(valid)
        print(f"[naive_llm] Parsed pairs: {len(valid)}/{len(rows)}  Accuracy: {acc:.1%}")
        print(
            f"[naive_llm] Metrics: overall={m['overall']:.3f} auc={m['auc']:.3f} "
            f"c@1={m['c@1']:.3f} F1={m['F1']:.3f} brier={m['brier']:.3f}"
        )


if __name__ == "__main__":
    main()
