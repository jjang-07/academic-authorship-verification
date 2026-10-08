"""End-to-end smoke test for the analyst agent pipeline.

Runs the full pipeline on a single Reuters text pair:

    load_reuters → extract_all → format_evidence_packet → AnalystAgent.analyze

Usage (from project root)
--------------------------
    python tests/test_analyst.py

    # Use a specific model:
    ANALYST_MODEL=gpt-4o python tests/test_analyst.py

    # Dry-run: skip the API call and just print the evidence packet
    python tests/test_analyst.py --dry-run

Prerequisites
-------------
- config.REUTERS_DIR must point to a directory containing at least one author
  subdirectory with two or more .txt files.
- OPENAI_API_KEY (or ANTHROPIC_API_KEY) must be set in .env when not using
  --dry-run.
- spaCy model ``en_core_web_md`` must be installed:
      python -m spacy download en_core_web_md

Exit codes
----------
0  success (including dry-run)
1  pipeline error (missing data, extraction failure, etc.)
2  API call failed (key missing, network error, etc.)
"""

from __future__ import annotations

import argparse
import sys
import os
import textwrap
import traceback

# ── Make sure the project root is on sys.path when this script is run directly
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402  (must come after sys.path fix)
from data.preprocessor import load_reuters  # noqa: E402
from data.load_cefr import get_cefr_dict  # noqa: E402
from features.handcrafted import extract_all  # noqa: E402
from features.evidence_packet import format_evidence_packet  # noqa: E402
from agents.analyst_agent import AnalystAgent  # noqa: E402


# ──────────────────────────────── Helpers ─────────────────────────────────────

def _banner(title: str, width: int = 60) -> str:
    return f"\n{'═' * width}\n{title.center(width)}\n{'═' * width}"


def _section(title: str) -> str:
    return f"\n{'─' * 60}\n{title}\n{'─' * 60}"


def _wrap(text: str, indent: int = 2) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=78, initial_indent=prefix,
                         subsequent_indent=prefix)


# ──────────────────────────────── Main ────────────────────────────────────────

def run(dry_run: bool = False, pair_index: int = 0) -> int:
    """Execute the end-to-end pipeline.  Returns an exit code (0 = ok)."""

    print(_banner("V2 ANALYST PIPELINE — SMOKE TEST"))

    # ── Step 1: Load a Reuters pair ───────────────────────────────────────────
    print(_section("STEP 1 · Load Reuters pair"))
    print(f"  Reuters directory : {config.REUTERS_DIR}")

    try:
        pairs = load_reuters(
            base_dir=config.REUTERS_DIR,
            seed=config.RANDOM_SEED,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ERROR loading Reuters corpus: {exc}")
        print("  Check that config.REUTERS_DIR / REUTERS_DIR in .env is correct.")
        return 1

    if not pairs:
        print("\n  ERROR: load_reuters returned an empty list.")
        print(f"  Directory: {config.REUTERS_DIR}")
        print("  Check that the directory exists and contains author subdirectories.")
        return 1

    if pair_index >= len(pairs):
        print(f"\n  WARNING: pair_index={pair_index} exceeds {len(pairs)} pairs; "
              "using index 0.")
        pair_index = 0

    pair = pairs[pair_index]
    label_str = "SAME_AUTHOR" if pair.label == 1 else "DIFFERENT_AUTHOR"
    print(f"  Loaded {len(pairs)} pairs total. Using pair #{pair_index}.")
    print(f"  Author ID   : {pair.author_id}")
    print(f"  Ground truth: {label_str}")
    print(f"  Dataset     : {pair.source_dataset}")
    print(f"  Text A      : {len(pair.text_a):,} chars")
    print(f"  Text B      : {len(pair.text_b):,} chars")

    # ── Step 2: Load CEFR wordlist (optional) ─────────────────────────────────
    print(_section("STEP 2 · Load CEFR wordlist"))
    try:
        cefr_dict = get_cefr_dict()
        print(f"  Loaded {len(cefr_dict):,} CEFR entries from {config.CEFR_WORDLIST_PATH}")
    except FileNotFoundError as exc:
        print(f"  WARNING: {exc}")
        print("  CEFR features will be omitted from the evidence packet.")
        cefr_dict = None

    # ── Step 3: Extract features ───────────────────────────────────────────────
    print(_section("STEP 3 · Extract features  (spaCy parse — may take a moment)"))

    try:
        print("  Extracting features for Text A...")
        features_a = extract_all(pair.text_a, cefr_dict=cefr_dict)
        print(f"  → {len(features_a)} features extracted")

        print("  Extracting features for Text B...")
        features_b = extract_all(pair.text_b, cefr_dict=cefr_dict)
        print(f"  → {len(features_b)} features extracted")
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ERROR during feature extraction: {exc}")
        traceback.print_exc()
        return 1

    # Sample a few key values for visual confirmation
    key_sample = [
        "sent_mean", "sent_std", "vocab_ttr",
        "cefr_C1", "cefr_C2", "passive_voice_freq",
        "readability_flesch_kincaid_grade",
    ]
    print("\n  Feature spot-check:")
    for k in key_sample:
        va = features_a.get(k)
        vb = features_b.get(k)
        if va is not None or vb is not None:
            va_str = f"{va:.3f}" if va is not None else "n/a"
            vb_str = f"{vb:.3f}" if vb is not None else "n/a"
            print(f"    {k:<42} A={va_str}  B={vb_str}")

    # ── Step 4: Format evidence packet ────────────────────────────────────────
    print(_section("STEP 4 · Format evidence packet"))

    try:
        packet = format_evidence_packet(features_a, features_b)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ERROR formatting evidence packet: {exc}")
        traceback.print_exc()
        return 1

    print(f"  Evidence packet: {len(packet)} chars, "
          f"{packet.count(chr(10))} lines")
    print("\n  ── Packet preview (first 600 chars) ──")
    print(textwrap.indent(packet[:600], "  "))
    if len(packet) > 600:
        print("  [... truncated ...]")

    if dry_run:
        print("\n  [DRY RUN] Skipping API call.  Pipeline looks healthy.")
        print(_banner("SMOKE TEST PASSED (dry-run)"))
        return 0

    # ── Step 5: Call Analyst Agent ────────────────────────────────────────────
    print(_section("STEP 5 · Call Analyst Agent"))
    print(f"  Model       : {config.ANALYST_MODEL}")
    print(f"  Temperature : {config.AGENT_TEMPERATURE}")
    print("  Sending request to LLM API...")

    try:
        agent = AnalystAgent()
        response = agent.analyze(pair, packet)
    except (ValueError, ImportError) as exc:
        print(f"\n  ERROR (config / SDK): {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ERROR (API call): {exc}")
        traceback.print_exc()
        return 2

    # ── Step 6: Print result ──────────────────────────────────────────────────
    print(_section("ANALYST RESPONSE"))

    verdict_match = (response.verdict == label_str)
    match_tag = "✓ CORRECT" if verdict_match else "✗ WRONG"

    print(f"\n  Verdict      : {response.verdict}   [{match_tag}]")
    print(f"  Confidence   : {response.confidence:.2f}")
    print(f"  Ground truth : {label_str}")
    print(f"  Parse OK     : {response.parse_ok}")

    if response.key_features_cited:
        print(f"\n  Key features cited ({len(response.key_features_cited)}):")
        for feat in response.key_features_cited:
            print(f"    • {feat}")

    print("\n  Reasoning:")
    for para in response.reasoning.split("\n\n"):
        para = para.strip()
        if para:
            print(_wrap(para))
            print()

    if not response.parse_ok:
        print("\n  WARNING: parse_ok=False — the LLM did not follow the output format.")
        print("  Raw response (first 300 chars):")
        print(textwrap.indent(response.raw_response[:300], "    "))

    print(_banner("SMOKE TEST PASSED"))
    return 0


# ──────────────────────────────── Entry Point ─────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end smoke test for the V2 analyst pipeline.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all steps except the LLM API call. "
             "Useful for verifying preprocessing and feature extraction.",
    )
    parser.add_argument(
        "--pair",
        type=int,
        default=0,
        metavar="N",
        help="Index of the pair to use from the loaded Reuters pairs (default: 0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(run(dry_run=args.dry_run, pair_index=args.pair))
