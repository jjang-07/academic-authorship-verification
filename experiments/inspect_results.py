"""Inspect JSONL details from FULL_V2_TEXTONLY (or similar) runs.

Reads a JSONL details file and prints each pair with Analyst / Skeptic / Judge
sections, or a one-line summary per pair with --summary-only.

Usage
-----
    python experiments/inspect_results.py experiments/results/full_v2_textonly_details_*.jsonl

    python experiments/inspect_results.py path/to/details.jsonl --pair 42

    python experiments/inspect_results.py path/to/details.jsonl --wrong

    python experiments/inspect_results.py path/to/details.jsonl --summary-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BAR_WIDTH = 56
BAR = "═" * BAR_WIDTH


def _fmt(val) -> str:
    if val is None:
        return "(none)"
    return str(val)


def _fmt_conf(val) -> str:
    if val is None:
        return "(none)"
    if isinstance(val, (int, float)):
        return str(val)
    return str(val)


def _display_field(val) -> str:
    """Format list or scalar for Challenges / Overlooked / Decisive factors."""
    if val is None:
        return "(none)"
    if isinstance(val, list):
        if not val:
            return "(none)"
        return " | ".join(str(x).strip() for x in val if x is not None and str(x).strip())
    return str(val)


def _marker(record: dict) -> str:
    if "error" in record:
        return "⚠"
    c = record.get("correct")
    if c is True:
        return "✓"
    if c is False:
        return "✗"
    return "?"


def _header_conf(record: dict) -> str:
    """Confidence shown in the one-line header (judge final)."""
    jc = record.get("judge_confidence")
    if jc is not None:
        return _fmt_conf(jc)
    ac = record.get("analyst_confidence")
    if ac is not None:
        return _fmt_conf(ac)
    return "(none)"


def _format_pair_header(record: dict) -> str:
    pid = record.get("pair_id", "?")
    gt = _fmt(record.get("ground_truth"))
    fv = record.get("final_verdict") or record.get("judge_verdict")
    final_s = _fmt(fv) if fv is not None else "(none)"
    mk = _marker(record)
    conf = _header_conf(record)
    return f"Pair {pid} | Ground truth: {gt} | Final verdict: {final_s} | {mk} | Conf: {conf}"


def _format_pair_full(record: dict, *, show_raw: bool = False) -> str:
    lines: list[str] = [BAR, _format_pair_header(record), BAR]

    if record.get("error"):
        lines.append("")
        lines.append(f"ERROR: {_fmt(record['error'])}")
        lines.append("")

    av = _fmt(record.get("analyst_verdict"))
    ac = _fmt_conf(record.get("analyst_confidence"))
    lines.append(f"ANALYST:  [{av}] (conf=[{ac}])")
    lines.append(_wrap_block(record.get("analyst_reasoning")))
    lines.append("")

    ss = _fmt(record.get("skeptic_stance"))
    sc = _fmt_conf(record.get("skeptic_confidence"))
    lines.append(f"SKEPTIC:  [{ss}] (conf=[{sc}])")
    lines.append(f"Challenges: {_display_field(record.get('skeptic_challenges'))}")
    lines.append(f"Overlooked: {_display_field(record.get('skeptic_overlooked_evidence'))}")
    lines.append("Revised reasoning:")
    lines.append(_wrap_block(record.get("skeptic_revised_reasoning")))
    lines.append("")

    if record.get("analyst_rebuttal_verdict") is not None:
        rv = _fmt(record.get("analyst_rebuttal_verdict"))
        rc = _fmt_conf(record.get("analyst_rebuttal_confidence"))
        lines.append(f"ANALYST REBUTTAL (round 2):  [{rv}] (conf=[{rc}])")
        lines.append(_wrap_block(record.get("analyst_rebuttal_reasoning")))
        lines.append("")

    jv = _fmt(record.get("judge_verdict"))
    jc = _fmt_conf(record.get("judge_confidence"))
    lines.append(f"JUDGE:  [{jv}] (conf=[{jc}])")
    aa = record.get("judge_agent_agreement")
    if aa is not None:
        lines.append(f"Agent agreement: {_fmt(aa)}")
    lines.append(f"Decisive factors: {_display_field(record.get('judge_decisive_factors'))}")
    lines.append("Educator summary:")
    lines.append(_wrap_block(record.get("judge_educator_summary")))
    lines.append("")

    el = record.get("elapsed_seconds")
    if el is not None:
        lines.append(f"Elapsed: {el}s")
    else:
        lines.append("Elapsed: (none)")

    if show_raw:
        lines.append("")
        lines.append("── RAW MODEL OUTPUTS (verbatim) ──")
        for title, key in (
            ("ANALYST", "analyst_raw_response"),
            ("SKEPTIC", "skeptic_raw_response"),
            ("ANALYST REBUTTAL", "analyst_rebuttal_raw_response"),
            ("JUDGE", "judge_raw_response"),
        ):
            raw = record.get(key)
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                continue
            lines.append("")
            lines.append(f"{title}:")
            lines.append(_wrap_block(raw))

    lines.append("")
    return "\n".join(lines)


def _wrap_block(text, width: int = 72, indent: int = 0) -> str:
    """Wrap analyst reasoning (may be long, multi-paragraph)."""
    if text is None:
        return " " * indent + "(none)"
    s = str(text).strip()
    if not s:
        return " " * indent + "(none)"
    prefix = " " * indent
    max_len = width - indent
    out: list[str] = []
    for para in s.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        words = para.split()
        cur: list[str] = []
        cur_len = 0
        for w in words:
            add = len(w) + (1 if cur else 0)
            if cur_len + add <= max_len:
                cur.append(w)
                cur_len += add
            else:
                if cur:
                    out.append(prefix + " ".join(cur))
                cur = [w]
                cur_len = len(w)
        if cur:
            out.append(prefix + " ".join(cur))
    return "\n".join(out) if out else prefix + "(none)"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect JSONL details file from FULL_V2_TEXTONLY runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "jsonl",
        type=Path,
        help="Path to the JSONL details file (e.g. full_v2_textonly_details_*.jsonl).",
    )
    parser.add_argument(
        "--pair",
        type=int,
        metavar="N",
        help="Show only this pair ID.",
    )
    parser.add_argument(
        "--wrong",
        action="store_true",
        help="Show only incorrect predictions (and error rows).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the one-line header per pair (no analyst reasoning or long text).",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help=(
            "Append raw API completion text (analyst_raw_response, skeptic_raw_response, "
            "judge_raw_response, rebuttal raw if present) after each pair."
        ),
    )
    args = parser.parse_args()

    if not args.jsonl.exists():
        print(f"Error: file not found: {args.jsonl}", file=sys.stderr)
        return 1

    records: list[dict] = []
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipped malformed line: {e}", file=sys.stderr)

    if args.pair is not None:
        records = [r for r in records if r.get("pair_id") == args.pair]
        if not records:
            print(f"No pair with ID {args.pair} in {args.jsonl}", file=sys.stderr)
            return 1
    if args.wrong:
        records = [r for r in records if r.get("correct") is False or "error" in r]

    if not records:
        if args.wrong:
            print("No incorrect predictions in this file.")
        else:
            print("No records to display.")
        return 0

    print(f"\nShowing {len(records)} pair(s) from {args.jsonl}\n")
    if args.summary_only:
        print(BAR)
        for r in records:
            print(_format_pair_header(r))
        print(BAR)
        print("")
    else:
        for r in records:
            print(_format_pair_full(r, show_raw=args.show_raw))
    return 0


if __name__ == "__main__":
    sys.exit(main())
