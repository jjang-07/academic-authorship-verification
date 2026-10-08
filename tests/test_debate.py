"""End-to-end smoke test for the full three-agent debate pipeline.

Runs the complete V2 pipeline on a single Reuters text pair:

    load_reuters → extract_all → format_evidence_packet
        → AnalystAgent → SkepticAgent → JudgeAgent → DebateResult

All three agent outputs and the final verdict are printed in a structured
format suitable for inspection during development.

Usage (from project root)
--------------------------
    # Full 1-round debate (requires API key in .env):
    python tests/test_debate.py

    # 2-round debate with Analyst rebuttal:
    python tests/test_debate.py --rounds 2

    # Dry-run: skip API calls, just verify preprocessing + feature extraction:
    python tests/test_debate.py --dry-run

    # Use a different pair from the Reuters corpus (unshuffled list from load_reuters):
    python tests/test_debate.py --pair 3

    # Match pair_id from experiments/results/*.jsonl (shuffled + clipped like run_experiment):
    python tests/test_debate.py --text-only --pair-ids 17 43 --seed 42 --experiment-max-pairs 100
    # Same, plus full Analyst / Skeptic / Judge outputs after each pair (--summary):
    python tests/test_debate.py --text-only --pair-ids 17 43 --summary
    python tests/test_debate.py --gpt5 --dataset student --pairs 0 1 2 --summary

    # Print author_id, topics, and text lengths for each JSONL row (maps pair_id → Reuters pair):
    python tests/test_debate.py --print-jsonl-index experiments/results/full_v2_textonly_details_*.jsonl

    # Run multiple pairs sequentially with summary table:
    python tests/test_debate.py --gpt5 --pairs 0 1 2 3 4

    # Use the ICL analyst (raw text + 5-feature summary, no delta packet):
    python tests/test_debate.py --icl

    # Use the fully text-only pipeline (all three agents: close reading, no features):
    python tests/test_debate.py --text-only

    # Use gpt-5.4 via Responses API with reasoning=high on the text-only pipeline:
    python tests/test_debate.py --gpt5
    python tests/test_debate.py --gpt5 --pair 3 --rounds 2

    # Student essays (data/raw/student_essays); --pair / --pairs index like Reuters:
    python tests/test_debate.py --dataset student --dry-run
    python tests/test_debate.py --dataset student --gpt5 --pairs 0 1

    # Combine flags:
    python tests/test_debate.py --icl --pair 2 --rounds 1
    python tests/test_debate.py --text-only --pair 2 --rounds 1

    # Use a specific model for all agents (overrides .env):
    ANALYST_MODEL=gpt-4o SKEPTIC_MODEL=gpt-4o JUDGE_MODEL=gpt-4o python tests/test_debate.py

Prerequisites
-------------
- config.REUTERS_DIR must resolve to the Reuters-50-50 directory.
- OPENAI_API_KEY (or ANTHROPIC_API_KEY) must be set in .env for live runs.
- spaCy model en_core_web_md must be installed:
      python -m spacy download en_core_web_md

Exit codes
----------
0  success (including dry-run)
1  pipeline error (data, feature extraction, etc.)
2  API / configuration error
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import textwrap
import traceback
from typing import List, Optional

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from data.preprocessor import TextPair, load_reuters  # noqa: E402
from data.student_essay_loader import (  # noqa: E402
    DEFAULT_BASE_DIR as _STUDENT_ESSAYS_DIR,
    load_student_essay_pairs,
)
from data.load_cefr import get_cefr_dict  # noqa: E402
from features.handcrafted import extract_all  # noqa: E402
from features.evidence_packet import format_evidence_packet  # noqa: E402
from agents.analyst_agent import AnalystAgent  # noqa: E402
from agents.analyst_agent_icl import AnalystAgentICL  # noqa: E402
from agents.analyst_agent_textonly import AnalystAgentTextOnly  # noqa: E402
from agents.skeptic_agent import SkepticAgent  # noqa: E402
from agents.skeptic_agent_textonly import SkepticAgentTextOnly  # noqa: E402
from agents.judge_agent import JudgeAgent  # noqa: E402
from agents.judge_agent_textonly import JudgeAgentTextOnly  # noqa: E402
from debate.orchestrator import DebateOrchestrator, DebateResult  # noqa: E402
from debate.orchestrator_textonly import DebateOrchestratorTextOnly  # noqa: E402
from debate.textonly_dataset_prompt import (  # noqa: E402
    DATASET_CONTEXT_STUDENT_ESSAYS,
    DATASET_CONTEXT_WIRE_SERVICE,
)
from agents.base_agent import AgentResponse  # noqa: E402
from agents.judge_agent import JudgeResponse  # noqa: E402
from agents.skeptic_agent import SkepticResponse  # noqa: E402


# ──────────────────────────────── Pair list (JSONL / run_experiment alignment) ─

def _experiment_ordered_pairs(
    seed: int,
    max_pairs: Optional[int] = 100,
) -> List[TextPair]:
    """Same ordering as ``run_experiment.py`` → ``run_full_v2_textonly`` JSONL ``pair_id``.

    1. ``load_reuters(..., seed=seed)``
    2. ``random.Random(seed).shuffle`` copy of that list
    3. optional clip to first ``max_pairs`` (default 100, matching ``--max-pairs``)

    JSONL ``pair_id`` is the 0-based index into the clipped shuffled list.
    """
    pairs = load_reuters(base_dir=config.REUTERS_DIR, seed=seed)
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    if max_pairs is not None:
        shuffled = shuffled[:max_pairs]
    return shuffled


def _student_experiment_ordered_pairs(
    seed: int,
    max_pairs: Optional[int] = 100,
) -> List[TextPair]:
    """Same ordering as ``run_student_experiment.py`` JSONL ``pair_id``.

    1. ``load_student_essay_pairs(..., seed=seed)``
    2. ``random.Random(seed).shuffle`` copy of that list
    3. optional clip to first ``max_pairs`` (``None`` = no clip when max_pairs unset)
    """
    pairs = load_student_essay_pairs(seed=seed, print_summary=False)
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    if max_pairs is not None:
        shuffled = shuffled[:max_pairs]
    return shuffled


def _print_jsonl_pair_index(
    jsonl_path: str,
    seed: int,
    max_pairs: Optional[int],
    *,
    dataset: str = "reuters",
) -> None:
    """Print pair_id → corpus metadata so JSONL rows can be located."""
    if dataset == "student":
        pairs_list = _student_experiment_ordered_pairs(seed, max_pairs)
    else:
        pairs_list = _experiment_ordered_pairs(seed, max_pairs)
    print(
        "pair_id\tauthor_id\ttopic_a\ttopic_b\tchars_a\tchars_b\tground_truth\t"
        "jsonl_line"
    )
    with open(jsonl_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"# line {lineno}: JSON error: {exc}")
                continue
            pid = rec.get("pair_id")
            if pid is None:
                print(f"# line {lineno}: missing pair_id")
                continue
            if not isinstance(pid, int) or pid < 0 or pid >= len(pairs_list):
                print(
                    f"# line {lineno}: pair_id={pid} out of range "
                    f"[0, {len(pairs_list)}); try --dataset / --experiment-max-pairs"
                )
                continue
            p = pairs_list[pid]
            gt = "SAME_AUTHOR" if p.label == 1 else "DIFFERENT_AUTHOR"
            print(
                f"{pid}\t{p.author_id}\t{p.topic_a}\t{p.topic_b}\t"
                f"{len(p.text_a)}\t{len(p.text_b)}\t{gt}\t{lineno}"
            )


# ──────────────────────────────── Display helpers ─────────────────────────────

def _banner(title: str, width: int = 64) -> str:
    return f"\n{'═' * width}\n{title.center(width)}\n{'═' * width}"


def _section(title: str, width: int = 64) -> str:
    return f"\n{'─' * width}\n{title}\n{'─' * width}"


def _wrap(text: str, indent: int = 2) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=80, initial_indent=prefix,
                         subsequent_indent=prefix)


def _print_bullets(items: list, indent: int = 4) -> None:
    prefix = " " * indent
    for item in items:
        # Re-wrap long bullet items cleanly
        wrapped = textwrap.fill(
            item, width=80,
            initial_indent=f"{prefix}• ",
            subsequent_indent=f"{prefix}  ",
        )
        print(wrapped)


def _verdict_tag(verdict: str, ground_truth: str) -> str:
    if verdict in ("SAME_AUTHOR", "DIFFERENT_AUTHOR"):
        return "✓ CORRECT" if verdict == ground_truth else "✗ WRONG"
    return "? UNKNOWN"


def _print_pair_debate_summary(
    pair_idx: int,
    result: DebateResult,
    *,
    jsonl_label: bool,
) -> None:
    """After each pair in --summary mode: Analyst, optional rebuttal, Skeptic, Judge."""
    pid = f"pair_id {pair_idx}" if jsonl_label else f"index {pair_idx}"
    pid_u = pid.upper()

    a: AgentResponse = result.analyst
    print(_section(f"ANALYST · {pid_u}"))
    print(f"  VERDICT     : {a.verdict}")
    print(f"  CONFIDENCE  : {a.confidence:.2f}")
    print(f"  Parse OK    : {a.parse_ok}")
    if a.key_features_cited:
        print(f"\n  Key features ({len(a.key_features_cited)}):")
        _print_bullets(a.key_features_cited)
    if a.reasoning and a.reasoning.strip():
        print("\n  Reasoning:")
        for para in a.reasoning.split("\n\n"):
            para = para.strip()
            if para:
                print(_wrap(para))
                print()
    print("\n  Full Analyst output (verbatim):")
    print(textwrap.indent(a.raw_response.rstrip() or "(empty)", "    "))

    if result.rebuttal is not None:
        rb: AgentResponse = result.rebuttal
        print(_section(f"ANALYST REBUTTAL (ROUND 2) · {pid_u}"))
        print(f"  VERDICT     : {rb.verdict}")
        print(f"  CONFIDENCE  : {rb.confidence:.2f}")
        print(f"  Parse OK    : {rb.parse_ok}")
        if rb.key_features_cited:
            print(f"\n  Key features ({len(rb.key_features_cited)}):")
            _print_bullets(rb.key_features_cited)
        if rb.reasoning and rb.reasoning.strip():
            print("\n  Rebuttal reasoning:")
            for para in rb.reasoning.split("\n\n"):
                para = para.strip()
                if para:
                    print(_wrap(para))
                    print()
        print("\n  Full rebuttal output (verbatim):")
        print(textwrap.indent(rb.raw_response.rstrip() or "(empty)", "    "))

    s: SkepticResponse = result.skeptic
    print(_section(f"SKEPTIC · {pid_u}"))
    print(f"  STANCE      : {s.stance}")
    print(f"  CONFIDENCE  : {s.confidence:.2f}")
    print(f"  Parse OK    : {s.parse_ok}")
    if s.challenges:
        print("\n  Challenges:")
        _print_bullets(s.challenges)
    if s.overlooked_evidence:
        print("\n  Overlooked evidence:")
        _print_bullets(s.overlooked_evidence)
    if s.revised_reasoning and s.revised_reasoning.strip():
        print("\n  Revised reasoning:")
        for para in s.revised_reasoning.split("\n\n"):
            para = para.strip()
            if para:
                print(_wrap(para))
                print()
    print("\n  Full Skeptic output (verbatim):")
    print(textwrap.indent(s.raw_response.rstrip() or "(empty)", "    "))

    j: JudgeResponse = result.judge
    print(_section(f"JUDGE · {pid_u}"))
    print(f"  FINAL VERDICT   : {j.verdict}")
    print(f"  CONFIDENCE      : {j.confidence:.2f}")
    print(f"  Agent agreement : {j.agent_agreement}")
    print(f"  Parse OK        : {j.parse_ok}")
    if j.decisive_factors:
        print("\n  Decisive factors:")
        _print_bullets(j.decisive_factors)
    else:
        print("\n  Decisive factors: (none parsed)")
    if j.educator_summary:
        print("\n  Educator summary (plain-English reasoning):")
        print(_wrap(j.educator_summary.strip(), indent=4))
    else:
        print("\n  Educator summary: (empty)")
    print("\n  Full Judge output (verbatim):")
    print(textwrap.indent(j.raw_response.rstrip() or "(empty)", "    "))


# ──────────────────────────────── Pipeline steps 1–4 ─────────────────────────

def _load_pipeline(
    pairs_list: List[TextPair],
    pair_index: int,
    quiet: bool = False,
    *,
    pair_id_label: Optional[str] = None,
    dataset: str = "reuters",
):
    """Steps 1–4: take pair from ``pairs_list``, CEFR, extract features, format packet.

    Returns (pair, features_a, features_b, packet) or raises SystemExit.
    When quiet=True, suppress step-by-step output (for multi-pair runs).

    ``pair_id_label`` — if set, shown instead of raw index (e.g. JSONL ``pair_id``).
    """
    def _log(msg: str = "") -> None:
        if not quiet:
            print(msg)

    if dataset == "student":
        _log(_section("STEP 1 · Load text pair (student essays)"))
        _log(f"  Corpus root: {_STUDENT_ESSAYS_DIR}")
    else:
        _log(_section("STEP 1 · Load Reuters pair"))
        _log(f"  Directory  : {config.REUTERS_DIR}")
    if not pairs_list:
        print("\n  ERROR: pair list is empty.")
        sys.exit(1)

    if pair_index < 0 or pair_index >= len(pairs_list):
        print(
            f"\n  ERROR: pair_index={pair_index} out of range "
            f"[0, {len(pairs_list)}). For JSONL pair_id, check --seed, "
            f"--experiment-max-pairs (try 0 = no clip), and corpus path."
        )
        sys.exit(1)

    pair = pairs_list[pair_index]
    label_str = "SAME_AUTHOR" if pair.label == 1 else "DIFFERENT_AUTHOR"
    idx_disp = pair_id_label if pair_id_label is not None else str(pair_index)
    _log(f"  Pairs total : {len(pairs_list)}")
    _log(f"  Using pair  : #{idx_disp} (index {pair_index})")
    _log(f"  Author ID   : {pair.author_id}")
    _log(f"  Ground truth: {label_str}")
    _log(f"  Text A      : {len(pair.text_a):,} chars  |  Text B: {len(pair.text_b):,} chars")

    # Step 2 — CEFR
    _log(_section("STEP 2 · CEFR wordlist"))
    try:
        cefr_dict = get_cefr_dict()
        _log(f"  Loaded {len(cefr_dict):,} CEFR entries")
    except FileNotFoundError:
        _log("  WARNING: CEFR wordlist not found — CEFR features will be omitted.")
        cefr_dict = None

    # Step 3 — Feature extraction
    _log(_section("STEP 3 · Feature extraction  (spaCy — may take a moment)"))
    try:
        _log("  Text A...")
        features_a = extract_all(pair.text_a, cefr_dict=cefr_dict)
        _log(f"  → {len(features_a)} features")
        _log("  Text B...")
        features_b = extract_all(pair.text_b, cefr_dict=cefr_dict)
        _log(f"  → {len(features_b)} features")
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)

    # Step 4 — Evidence packet
    _log(_section("STEP 4 · Format evidence packet"))
    try:
        packet = format_evidence_packet(features_a, features_b)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)

    _log(f"  {len(packet):,} chars, {packet.count(chr(10))} lines")
    _log("\n  Preview (first 500 chars):")
    _log(textwrap.indent(packet[:500], "    "))
    if len(packet) > 500:
        _log("    [...]")

    return pair, features_a, features_b, packet


# ──────────────────────────────── Main ────────────────────────────────────────

def _run_icl_debate(
    pair,
    features_a: dict,
    features_b: dict,
    packet: str,
    rounds: int,
) -> "DebateResult":
    """Run the three-agent debate with the ICL analyst.

    The ICL analyst receives raw text + a five-feature summary instead of the
    structured evidence packet, so it cannot be dropped straight into
    DebateOrchestrator (which calls ``analyst.analyze(pair, packet)``).
    Skeptic and Judge are unchanged and still receive the full evidence packet.
    """
    import time

    t0 = time.perf_counter()

    # Agent A — ICL analyst
    icl_analyst = AnalystAgentICL()
    analyst_response = icl_analyst.analyze(pair, features_a, features_b)

    # Agent B — standard Skeptic (receives full evidence packet as usual)
    skeptic = SkepticAgent()
    skeptic_response = skeptic.challenge(pair, packet, analyst_response)

    # Optional round-2 rebuttal — ICL analyst re-uses its base ``call()`` method
    # (same format as standard rebuttal path in DebateOrchestrator)
    rebuttal = None
    if rounds >= 2:
        from debate.orchestrator import _build_rebuttal_message
        rebuttal_msg = _build_rebuttal_message(
            pair, packet, analyst_response, skeptic_response
        )
        rebuttal = icl_analyst.call(rebuttal_msg)

    # Agent C — standard Judge
    judge = JudgeAgent()
    judge_response = judge.adjudicate(
        pair, packet, analyst_response, skeptic_response, rebuttal
    )

    elapsed = time.perf_counter() - t0

    return DebateResult(
        analyst=analyst_response,
        skeptic=skeptic_response,
        judge=judge_response,
        rebuttal=rebuttal,
        final_verdict=judge_response.verdict,
        final_confidence=judge_response.confidence,
        elapsed_seconds=elapsed,
    )


_TEXTONLY_ANALYST_MODEL = "gpt-4o"         # pinned model for --text-only

# --gpt5 heterogeneous configuration
_GPT5_ANALYST_MODEL = "gpt-5.4"          # OpenAI Responses API, reasoning_effort=high
_GPT5_SKEPTIC_MODEL = "claude-opus-4-6"  # Anthropic API
_GPT5_JUDGE_MODEL   = "gpt-5.4"        # OpenAI Responses API, reasoning_effort=high


def _run_textonly_debate(
    pair,
    rounds: int,
    analyst_model: str = _TEXTONLY_ANALYST_MODEL,
    skeptic_model: Optional[str] = None,
    judge_model: Optional[str] = None,
    verbose: bool = True,
    dataset_context: Optional[str] = None,
) -> "DebateResult":
    """Run the fully text-only debate via DebateOrchestratorTextOnly.

    All three agents receive only raw text excerpts and the debate transcript —
    no evidence packet, no numeric deltas anywhere in the pipeline.

    Parameters
    ----------
    analyst_model : Model name for the Analyst.  Defaults to
                    ``_TEXTONLY_ANALYST_MODEL`` (gpt-4o).
    skeptic_model : Model name for the Skeptic.  ``None`` → uses
                    ``config.SKEPTIC_MODEL`` from .env.
    judge_model   : Model name for the Judge.  ``None`` → uses
                    ``config.JUDGE_MODEL`` from .env.

    Routing is automatic: ``gpt-5*`` → Responses API, ``claude-*`` → Anthropic.
    """
    analyst = AnalystAgentTextOnly(model=analyst_model)
    skeptic = SkepticAgentTextOnly(
        **({"model": skeptic_model} if skeptic_model else {})
    )
    judge = JudgeAgentTextOnly(
        **({"model": judge_model} if judge_model else {})
    )
    orchestrator = DebateOrchestratorTextOnly(
        analyst,
        skeptic,
        judge,
        max_rounds=rounds,
        verbose=verbose,
        dataset_context=dataset_context,
    )
    return orchestrator.run(pair)


def _run_single_debate(
    pair,
    features_a: dict,
    features_b: dict,
    packet: str,
    rounds: int,
    use_gpt5: bool,
    use_textonly: bool,
    use_icl: bool,
    verbose: bool = True,
    dataset: str = "reuters",
) -> "DebateResult":
    """Run one debate and return the result."""
    dctx = DATASET_CONTEXT_STUDENT_ESSAYS if dataset == "student" else None
    if use_gpt5:
        return _run_textonly_debate(
            pair, rounds,
            analyst_model=_GPT5_ANALYST_MODEL,
            skeptic_model=_GPT5_SKEPTIC_MODEL,
            judge_model=_GPT5_JUDGE_MODEL,
            verbose=verbose,
            dataset_context=dctx,
        )
    if use_textonly:
        return _run_textonly_debate(
            pair, rounds, verbose=verbose, dataset_context=dctx
        )
    if use_icl:
        return _run_icl_debate(pair, features_a, features_b, packet, rounds)
    analyst = AnalystAgent()
    skeptic = SkepticAgent()
    judge = JudgeAgent()
    orchestrator = DebateOrchestrator(
        analyst, skeptic, judge, max_rounds=rounds, verbose=verbose
    )
    return orchestrator.run(pair, packet)


def _run_multi_pairs(
    indices: list[int],
    rounds: int,
    use_gpt5: bool,
    use_textonly: bool,
    use_icl: bool,
    pairs_list: List[TextPair],
    *,
    jsonl_pair_ids: bool = False,
    summary_pair_detail: bool = False,
    dataset: str = "reuters",
) -> int:
    """Run debates for multiple pairs sequentially; print summary table."""
    import time as _time

    rows: list[dict] = []
    for i, pair_idx in enumerate(indices):
        pid_note = f" [JSONL pair_id]" if jsonl_pair_ids else ""
        print(f"\n  Pair {pair_idx}{pid_note} ({i + 1}/{len(indices)})...", end=" ", flush=True)
        t0 = _time.perf_counter()
        try:
            pair, features_a, features_b, packet = _load_pipeline(
                pairs_list,
                pair_idx,
                quiet=True,
                pair_id_label=str(pair_idx) if jsonl_pair_ids else None,
                dataset=dataset,
            )
            result = _run_single_debate(
                pair, features_a, features_b, packet, rounds,
                use_gpt5, use_textonly, use_icl, verbose=False,
                dataset=dataset,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}")
            traceback.print_exc()
            return 2
        elapsed = _time.perf_counter() - t0
        gt = "SAME_AUTHOR" if pair.label == 1 else "DIFFERENT_AUTHOR"
        correct = result.judge.verdict == gt
        tag = "✓" if correct else "✗"
        print(f"{tag} {result.judge.verdict} ({elapsed:.1f}s)")
        if summary_pair_detail:
            _print_pair_debate_summary(pair_idx, result, jsonl_label=jsonl_pair_ids)
        rows.append({
            "pair": pair_idx,
            "ground_truth": gt,
            "analyst": result.analyst.verdict,
            "skeptic": result.skeptic.stance,
            "judge": result.judge.verdict,
            "correct": correct,
            "elapsed": elapsed,
        })

    # Summary table
    col_w = {"pair": 6, "ground_truth": 18, "analyst": 18, "skeptic": 18, "judge": 18, "result": 12, "time": 10}
    sep = "  "
    header = (
        f"{'Pair':<{col_w['pair']}}{sep}"
        f"{'Ground Truth':<{col_w['ground_truth']}}{sep}"
        f"{'Analyst':<{col_w['analyst']}}{sep}"
        f"{'Skeptic':<{col_w['skeptic']}}{sep}"
        f"{'Judge':<{col_w['judge']}}{sep}"
        f"{'Result':<{col_w['result']}}{sep}"
        f"{'Time (s)':<{col_w['time']}}"
    )
    print(_section("SUMMARY TABLE"))
    print(header)
    print("─" * len(header))
    for r in rows:
        result_str = "✓ correct" if r["correct"] else "✗ incorrect"
        print(
            f"{r['pair']:<{col_w['pair']}}{sep}"
            f"{r['ground_truth']:<{col_w['ground_truth']}}{sep}"
            f"{r['analyst']:<{col_w['analyst']}}{sep}"
            f"{r['skeptic']:<{col_w['skeptic']}}{sep}"
            f"{r['judge']:<{col_w['judge']}}{sep}"
            f"{result_str:<{col_w['result']}}{sep}"
            f"{r['elapsed']:.1f}"
        )
    n_correct = sum(1 for r in rows if r["correct"])
    total_time = sum(r["elapsed"] for r in rows)
    print("─" * len(header))
    print(f"  Accuracy: {n_correct}/{len(rows)}  |  Total time: {total_time:.1f}s")
    print(_banner("SMOKE TEST PASSED"))
    return 0


def run(
    dry_run: bool = False,
    pair_index: int = 0,
    pair_indices: Optional[list[int]] = None,
    rounds: int = 1,
    use_icl: bool = False,
    use_textonly: bool = False,
    use_gpt5: bool = False,
    *,
    seed: Optional[int] = None,
    experiment_order: bool = False,
    experiment_max_pairs: Optional[int] = 100,
    pair_ids_from_jsonl: bool = False,
    pairs_list: Optional[List[TextPair]] = None,
    summary_pair_detail: bool = False,
    dataset: str = "reuters",
) -> int:
    seed = seed if seed is not None else config.RANDOM_SEED
    exp_clip = (
        None
        if experiment_max_pairs is None or experiment_max_pairs <= 0
        else experiment_max_pairs
    )

    if pairs_list is None:
        if dataset == "student":
            if experiment_order or pair_ids_from_jsonl:
                pairs_list = _student_experiment_ordered_pairs(seed, exp_clip)
            else:
                pairs_list = load_student_essay_pairs(seed=seed, print_summary=True)
        elif experiment_order or pair_ids_from_jsonl:
            pairs_list = _experiment_ordered_pairs(seed, exp_clip)
        else:
            pairs_list = load_reuters(base_dir=config.REUTERS_DIR, seed=seed)

    indices = pair_indices if pair_indices else [pair_index]
    multi_pair = len(indices) > 1

    if use_gpt5:
        analyst_label = "AnalystAgentTextOnly (close reading, no features)"
    elif use_textonly:
        analyst_label = "AnalystAgentTextOnly (close reading, no features)"
    elif use_icl:
        analyst_label = "AnalystAgentICL (raw text + 5-feature summary)"
    else:
        analyst_label = "AnalystAgent (evidence packet)"

    print(_banner("V2 DEBATE PIPELINE — SMOKE TEST"))
    print(f"  Analyst       : {analyst_label}")
    if use_gpt5:
        print(f"  Analyst model : {_GPT5_ANALYST_MODEL}  (OpenAI Responses API · reasoning_effort=high)")
        print(f"  Skeptic model : {_GPT5_SKEPTIC_MODEL}  (Anthropic API)")
        print(f"  Judge model   : {_GPT5_JUDGE_MODEL}  (OpenAI Responses API · reasoning_effort=high)")
    elif use_textonly:
        print(f"  Analyst model : {_TEXTONLY_ANALYST_MODEL} (pinned override)")
        print(f"  Skeptic model : {config.SKEPTIC_MODEL}")
        print(f"  Judge model   : {config.JUDGE_MODEL}")
    else:
        print(f"  Analyst model : {config.ANALYST_MODEL}")
        print(f"  Skeptic model : {config.SKEPTIC_MODEL}")
        print(f"  Judge model   : {config.JUDGE_MODEL}")
    print(f"  Rounds        : {rounds}")
    print(f"  Dataset       : {dataset}")
    if use_gpt5 or use_textonly:
        dctx = (
            DATASET_CONTEXT_STUDENT_ESSAYS
            if dataset == "student"
            else DATASET_CONTEXT_WIRE_SERVICE
        )
        print(f"  Dataset context: {dctx}")
    print(f"  Corpus seed   : {seed}")
    if experiment_order or pair_ids_from_jsonl:
        clip_disp = exp_clip if exp_clip is not None else "all (no clip)"
        label = "student essays" if dataset == "student" else "Reuters"
        print(
            f"  Pair ordering : experiment / JSONL ({label}; shuffled + clip={clip_disp})"
            + ("  [pair_id = JSONL field]" if pair_ids_from_jsonl else "")
        )
    else:
        if dataset == "student":
            print("  Pair ordering : student essays (load_student_essay_pairs order)")
        else:
            print("  Pair ordering : raw load_reuters list (not JSONL order)")
    if multi_pair:
        print(f"  Pairs         : {indices}")

    if dry_run:
        pair, _, _, _ = _load_pipeline(
            pairs_list,
            indices[0],
            quiet=multi_pair,
            pair_id_label=str(indices[0]) if pair_ids_from_jsonl else None,
            dataset=dataset,
        )
        print("\n  [DRY RUN] Skipping API calls — pipeline looks healthy.")
        print(_banner("SMOKE TEST PASSED (dry-run)"))
        return 0

    # ── Multi-pair mode: run sequentially, print summary table ───────────────
    if multi_pair:
        return _run_multi_pairs(
            indices,
            rounds,
            use_gpt5,
            use_textonly,
            use_icl,
            pairs_list,
            jsonl_pair_ids=pair_ids_from_jsonl,
            summary_pair_detail=summary_pair_detail,
            dataset=dataset,
        )

    # ── Single-pair mode: full verbose output ───────────────────────────────────
    pair, features_a, features_b, packet = _load_pipeline(
        pairs_list,
        indices[0],
        pair_id_label=str(indices[0]) if pair_ids_from_jsonl else None,
        dataset=dataset,
    )
    label_str = "SAME_AUTHOR" if pair.label == 1 else "DIFFERENT_AUTHOR"

    if use_gpt5:
        mode_tag = (f"{_GPT5_ANALYST_MODEL} · {_GPT5_SKEPTIC_MODEL} · "
                    f"{_GPT5_JUDGE_MODEL}")
    elif use_textonly:
        mode_tag = "TEXT-ONLY ANALYST"
    elif use_icl:
        mode_tag = "ICL ANALYST"
    else:
        mode_tag = "standard"
    print(_section(f"STEP 5 · Three-agent debate ({mode_tag})"))
    try:
        result = _run_single_debate(
            pair, features_a, features_b, packet, rounds,
            use_gpt5, use_textonly, use_icl, verbose=True,
            dataset=dataset,
        )
    except (ValueError, ImportError) as exc:
        print(f"\n  ERROR (config / SDK): {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ERROR (API): {exc}")
        traceback.print_exc()
        return 2

    # ── Print all agent outputs ───────────────────────────────────────────────

    # Agent A — Analyst
    print(_section("AGENT A · ANALYST VERDICT"))
    a = result.analyst
    print(f"  Verdict      : {a.verdict}  [{_verdict_tag(a.verdict, label_str)}]")
    print(f"  Confidence   : {a.confidence:.2f}")
    print(f"  Ground truth : {label_str}")
    print(f"  Parse OK     : {a.parse_ok}")
    if a.key_features_cited:
        print(f"\n  Key features ({len(a.key_features_cited)}):")
        _print_bullets(a.key_features_cited)
    print("\n  Reasoning:")
    for para in a.reasoning.split("\n\n"):
        para = para.strip()
        if para:
            print(_wrap(para))
            print()
    if not a.parse_ok:
        print("  ⚠ parse_ok=False — raw (first 200 chars):")
        print(textwrap.indent(a.raw_response[:200], "    "))

    # Agent A — Rebuttal (Round 2 only)
    if result.rebuttal is not None:
        print(_section("AGENT A · REBUTTAL (Round 2)"))
        rb = result.rebuttal
        print(f"  Updated verdict     : {rb.verdict}")
        print(f"  Updated confidence  : {rb.confidence:.2f}")
        print(f"  Parse OK            : {rb.parse_ok}")
        print("\n  Rebuttal reasoning:")
        for para in rb.reasoning.split("\n\n"):
            para = para.strip()
            if para:
                print(_wrap(para))
                print()

    # Agent B — Skeptic
    print(_section("AGENT B · SKEPTIC CHALLENGE"))
    s = result.skeptic
    print(f"  Stance       : {s.stance}")
    print(f"  Confidence   : {s.confidence:.2f}")
    print(f"  Parse OK     : {s.parse_ok}")
    if s.challenges:
        print("\n  Challenges:")
        _print_bullets(s.challenges)
    if s.overlooked_evidence:
        print("\n  Overlooked evidence:")
        _print_bullets(s.overlooked_evidence)
    if s.revised_reasoning:
        print("\n  Revised reasoning:")
        for para in s.revised_reasoning.split("\n\n"):
            para = para.strip()
            if para:
                print(_wrap(para))
                print()
    if not s.parse_ok:
        print("  ⚠ parse_ok=False — raw (first 200 chars):")
        print(textwrap.indent(s.raw_response[:200], "    "))

    # Agent C — Judge
    print(_section("AGENT C · JUDGE FINAL VERDICT"))
    j = result.judge
    final_tag = _verdict_tag(j.verdict, label_str)
    print(f"  FINAL VERDICT  : {j.verdict}  [{final_tag}]")
    print(f"  CONFIDENCE     : {j.confidence:.2f}")
    print(f"  Agent agreement: {j.agent_agreement}")
    print(f"  Ground truth   : {label_str}")
    print(f"  Parse OK       : {j.parse_ok}")
    if j.decisive_factors:
        print("\n  Decisive factors:")
        _print_bullets(j.decisive_factors)
    if j.educator_summary:
        print("\n  Educator summary:")
        print(_wrap(j.educator_summary, indent=4))
    if not j.parse_ok:
        print("  ⚠ parse_ok=False — raw (first 200 chars):")
        print(textwrap.indent(j.raw_response[:200], "    "))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(_section("DEBATE SUMMARY"))
    print(f"  Ground truth   : {label_str}")
    print(f"  Analyst        : {a.verdict} (conf={a.confidence:.2f})")
    print(f"  Skeptic stance : {s.stance} (conf={s.confidence:.2f})")
    if result.rebuttal:
        print(f"  Rebuttal       : {result.rebuttal.verdict} (conf={result.rebuttal.confidence:.2f})")
    print(f"  Final verdict  : {j.verdict} (conf={j.confidence:.2f})  [{final_tag}]")
    print(f"  Elapsed        : {result.elapsed_seconds:.1f}s")

    print(_banner("SMOKE TEST PASSED"))
    return 0


# ──────────────────────────────── Entry point ─────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full three-agent debate smoke test for the V2 pipeline.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run preprocessing and feature extraction only; skip all API calls.",
    )
    parser.add_argument(
        "--dataset",
        choices=("reuters", "student"),
        default="reuters",
        help=(
            "Corpus: reuters (load_reuters) or student (load_student_essay_pairs from "
            "data/raw/student_essays). --pair / --pairs index into the same list rules "
            "as Reuters for that mode."
        ),
    )
    parser.add_argument(
        "--pair", type=int, default=0, metavar="N",
        help="Index into the pair list (default: 0). Ignored if --pairs is set.",
    )
    parser.add_argument(
        "--pairs", type=int, nargs="+", metavar="N",
        help="Run multiple pairs sequentially and print a summary table. E.g. --pairs 0 1 2 3 4.",
    )
    parser.add_argument(
        "--pair-ids",
        type=int,
        nargs="+",
        metavar="ID",
        help=(
            "Like --pairs but values are JSONL ``pair_id`` fields from experiment details "
            "(run_experiment FULL_V2_TEXTONLY for Reuters; run_student_experiment for "
            "--dataset student). Implies shuffled+clipped ordering; set --seed and "
            "--experiment-max-pairs to match that run."
        ),
    )
    parser.add_argument(
        "--experiment-order",
        action="store_true",
        help=(
            "Interpret --pair / --pairs as indices into the shuffled, optionally clipped "
            "list used by experiments/run_experiment.py (not raw load_reuters order)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Random seed for corpus load / pair generation + experiment shuffle "
            "(default: config.RANDOM_SEED)."
        ),
    )
    parser.add_argument(
        "--experiment-max-pairs",
        type=int,
        default=100,
        metavar="N",
        help=(
            "Clip shuffled list to N pairs (default 100 = run_experiment --max-pairs). "
            "Use 0 for no clip (full shuffled corpus)."
        ),
    )
    parser.add_argument(
        "--print-jsonl-index",
        metavar="PATH",
        help=(
            "Print a TSV mapping each JSONL line's pair_id to author_id, topics, "
            "text char lengths, and ground truth — then exit (no API calls). "
            "Use the same --dataset, --seed, and --experiment-max-pairs as the experiment run."
        ),
    )
    parser.add_argument(
        "--rounds", type=int, default=1, choices=[1, 2],
        help="Number of debate rounds (1 = standard, 2 = adds Analyst rebuttal).",
    )
    parser.add_argument(
        "--icl", action="store_true",
        help=(
            "Use the ICL Analyst (AnalystAgentICL) instead of the standard AnalystAgent. "
            "The ICL analyst receives raw text excerpts plus a five-feature natural-language "
            "summary instead of the structured delta-annotated evidence packet. "
            "Skeptic and Judge are unchanged."
        ),
    )
    parser.add_argument(
        "--text-only", action="store_true",
        help=(
            "Run the fully text-only pipeline (DebateOrchestratorTextOnly). "
            "All three agents — Analyst, Skeptic, and Judge — receive only the raw "
            "text excerpts and the debate transcript. No evidence packet, no numeric "
            "deltas, no severity labels anywhere in the pipeline. "
            "The Analyst is pinned to gpt-4o; Skeptic and Judge use config models. "
            "Cannot be combined with --icl or --gpt5."
        ),
    )
    parser.add_argument(
        "--gpt5", action="store_true",
        help=(
            f"Use the text-only pipeline with the Analyst pinned to {_GPT5_ANALYST_MODEL} "
            "via the OpenAI Responses API (/v1/responses) with reasoning={'effort': 'high'}. "
            "Skeptic and Judge use their configured models from config / .env. "
            "Implies --text-only prompt; cannot be combined with --icl or --text-only."
        ),
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help=(
            "With multiple pair indices (--pairs or --pair-ids): after each pair, print "
            "full Analyst (verdict, key features, reasoning, raw), Skeptic (stance, "
            "challenges, overlooked evidence, revised reasoning, raw), and Judge "
            "(decisive factors, educator summary, raw); plus Analyst rebuttal if --rounds 2. "
            "Ignored for single-pair runs (--pair only)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    exclusive = sum([args.icl, args.text_only, args.gpt5])
    if exclusive > 1:
        print("ERROR: --icl, --text-only, and --gpt5 are mutually exclusive. Pick one.")
        sys.exit(1)

    seed = args.seed if args.seed is not None else config.RANDOM_SEED
    exp_clip = None if args.experiment_max_pairs <= 0 else args.experiment_max_pairs

    if args.print_jsonl_index:
        _print_jsonl_pair_index(
            args.print_jsonl_index,
            seed,
            exp_clip,
            dataset=args.dataset,
        )
        sys.exit(0)

    if args.pair_ids is not None and args.pairs is not None:
        print("ERROR: use either --pair-ids or --pairs, not both.")
        sys.exit(1)

    if args.pair_ids is not None:
        run_indices = args.pair_ids
        pair_ids_from_jsonl = True
        experiment_order = True
    elif args.pairs is not None:
        run_indices = args.pairs
        pair_ids_from_jsonl = False
        experiment_order = args.experiment_order
    else:
        run_indices = [args.pair]
        pair_ids_from_jsonl = False
        experiment_order = args.experiment_order

    pair_indices_arg = run_indices if len(run_indices) > 1 else None
    summary_detail = bool(args.summary and pair_indices_arg is not None)
    if args.summary and pair_indices_arg is None:
        print(
            "WARNING: --summary only applies when 2+ pair indices are run "
            "(use e.g. --pairs 0 1 or --pair-ids 0 1); ignoring.",
            file=sys.stderr,
        )
    sys.exit(run(
        dry_run=args.dry_run,
        pair_index=run_indices[0],
        pair_indices=pair_indices_arg,
        rounds=args.rounds,
        use_icl=args.icl,
        use_textonly=args.text_only,
        use_gpt5=args.gpt5,
        seed=seed,
        experiment_order=experiment_order,
        experiment_max_pairs=args.experiment_max_pairs,
        pair_ids_from_jsonl=pair_ids_from_jsonl,
        summary_pair_detail=summary_detail,
        dataset=args.dataset,
    ))
