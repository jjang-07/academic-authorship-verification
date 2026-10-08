"""Diagnostic: call Analyst agent on pairs 0-4 and print raw LLM output.

Run from project root:
    python experiments/diagnose_analyst.py

Prints:
1. Full analyst_prompt.txt system prompt
2. For each pair 0-4: ground truth label, then raw response text before parsing
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from agents.analyst_agent import AnalystAgent
from data.preprocessor import load_reuters
from features.evidence_packet import format_evidence_packet
from features.handcrafted import extract_all

# Optional CEFR
def _get_cefr():
    try:
        from data.load_cefr import get_cefr_dict
        return get_cefr_dict()
    except Exception:
        return None


def main():
    # 1. Print full system prompt
    prompt_path = Path(__file__).parent.parent / "agents" / "prompts" / "analyst_prompt.txt"
    prompt_text = prompt_path.read_text(encoding="utf-8")
    print("=" * 70)
    print("FULL ANALYST SYSTEM PROMPT (analyst_prompt.txt)")
    print("=" * 70)
    print(prompt_text)
    print("=" * 70)
    print()

    # 2. Load pairs and CEFR
    pairs = load_reuters(config.REUTERS_DIR, seed=42)
    if len(pairs) < 5:
        print(f"ERROR: Need at least 5 pairs, got {len(pairs)}")
        sys.exit(1)
    cefr = _get_cefr()

    # 3. Build analyst and call _raw_call on pairs 0-4
    agent = AnalystAgent()

    for i in range(5):
        pair = pairs[i]
        fa = extract_all(pair.text_a, cefr_dict=cefr)
        fb = extract_all(pair.text_b, cefr_dict=cefr)
        packet = format_evidence_packet(fa, fb)

        # Build user message same way as AnalystAgent.analyze
        from agents.analyst_agent import _build_user_message
        user_msg = _build_user_message(pair, packet)

        print("=" * 70)
        print(f"PAIR {i}  |  Ground truth: {'SAME_AUTHOR' if pair.label else 'DIFFERENT_AUTHOR'}")
        print("=" * 70)
        print("RAW LLM RESPONSE (before parsing):")
        print("-" * 70)
        raw = agent._raw_call(user_msg)
        print(raw)
        print("-" * 70)
        print()


if __name__ == "__main__":
    main()
