"""Educator-facing authorship verification report formatter.

Takes a completed ``DebateResult`` (from the three-agent debate) plus an optional
``ShapResult`` (from SHAP analysis) and formats a clean, plain-English report
suitable for a teacher or administrator with no linguistics background.

Report sections
---------------
1. HEADER — verdict and confidence in everyday language
2. MOST IMPORTANT SIGNALS — top SHAP features with human-readable descriptions
   and plain-English interpretation of what each difference means
3. PLAIN ENGLISH SUMMARY — the Judge's ``educator_summary`` verbatim
4. DEBATE RECORD — which agent arguments were decisive (Judge's ``decisive_factors``)
5. AGENT CONSENSUS — whether agents agreed or disagreed
6. DISCLAIMER — this is a flagging tool for human review, not a final judgment

Output format matches the spec example:

    ═══════════════════════════════════════════
    AUTHORSHIP VERIFICATION REPORT
    ═══════════════════════════════════════════
    Verdict: DIFFERENT AUTHOR (Confidence: 82%)
    ...

Public API
----------
    report_str = format_report(debate_result, shap_result=shap_result)
    save_report(report_str, path="experiments/results/report_pair_0.txt")

    # Convenience: format and save in one call
    path = format_and_save(debate_result, shap_result, output_dir="experiments/results/")
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from debate.orchestrator import DebateResult


# ──────────────────────────────── Formatting Constants ────────────────────────

_BORDER = "═" * 47
_DIVIDER = "─" * 47
_LINE_WIDTH = 47       # wrapping width for body text
_INDENT = "  "


# ──────────────────────────────── Helpers ─────────────────────────────────────

def _wrap(text: str, indent: str = _INDENT, width: int = _LINE_WIDTH) -> str:
    """Wrap text to LINE_WIDTH with consistent indentation."""
    return textwrap.fill(text, width=width,
                         initial_indent=indent,
                         subsequent_indent=indent)


def _verdict_label(verdict: str) -> str:
    """Convert raw verdict string to display label."""
    v = verdict.upper()
    if v == "SAME_AUTHOR":
        return "SAME AUTHOR"
    if v == "DIFFERENT_AUTHOR":
        return "DIFFERENT AUTHOR"
    return "UNCERTAIN"


def _confidence_label(confidence: float) -> str:
    """Convert a float confidence to an everyday English descriptor."""
    c = max(0.0, min(1.0, confidence))
    if c >= 0.90:
        return "very strong"
    if c >= 0.80:
        return "strong"
    if c >= 0.70:
        return "moderate"
    if c >= 0.60:
        return "weak"
    return "very uncertain"


def _agreement_label(agent_agreement: str) -> str:
    """Convert FULL/PARTIAL/NONE to a human-readable phrase."""
    a = (agent_agreement or "").upper()
    if a == "FULL":
        return "All agents agreed."
    if a == "PARTIAL":
        return "Agents partially agreed (some disagreement)."
    if a == "NONE":
        return "Agents disagreed — verdict is based on the Judge's independent assessment."
    return "Agent agreement unknown."


def _shap_direction_phrase(direction: str) -> str:
    """Convert SHAP direction to a short English phrase."""
    if direction == "DIFFERENT_AUTHOR":
        return "suggests different authors"
    return "suggests same author"


# ──────────────────────────────── Section Builders ────────────────────────────

def _build_header(debate: DebateResult) -> str:
    """Section 1: Verdict headline."""
    label   = _verdict_label(debate.final_verdict)
    pct     = round(debate.final_confidence * 100)
    conf_l  = _confidence_label(debate.final_confidence)
    lines = [
        _BORDER,
        "AUTHORSHIP VERIFICATION REPORT",
        _BORDER,
        f"Verdict:    {label}",
        f"Confidence: {pct}% ({conf_l})",
        "",
    ]
    return "\n".join(lines)


def _build_shap_signals(shap_result) -> str:  # Optional[ShapResult]
    """Section 2: Top SHAP features with human-readable interpretations."""
    if shap_result is None or not shap_result.top_features:
        return ""

    lines = [
        "Most Important Signals (quantitative stylometric analysis):",
        _DIVIDER,
    ]
    for rank, feat in enumerate(shap_result.top_features, 1):
        direction_phrase = _shap_direction_phrase(feat.direction)
        delta_str = f"difference: {feat.delta:.3f}"
        shap_str  = f"weight: {abs(feat.shap_value):.3f}"
        # Plain-English interpretation line
        interp = _interpret_shap_feature(feat)
        lines.append(f"  {rank}. {feat.description.capitalize()}")
        lines.append(f"     ({delta_str}, {shap_str}) — {direction_phrase}")
        if interp:
            lines.append(f"     {interp}")

    lines.append("")
    return "\n".join(lines)


def _interpret_shap_feature(feat) -> str:  # ShapFeature
    """Generate a one-sentence plain-English interpretation for a SHAP feature."""
    name  = feat.name
    delta = feat.delta
    direc = feat.direction

    same    = direc == "SAME_AUTHOR"
    diff    = not same
    d_str   = f"{delta:.3f}"

    if "cefr_C1" in name or "cefr_C2" in name:
        level = "C1" if "C1" in name else "C2"
        return (
            "The two texts have similar rates of advanced vocabulary."
            if same else
            f"One text uses notably more {level}-level advanced vocabulary than the other."
        )
    if "vocab_ttr" in name:
        return (
            "Both texts show similar vocabulary diversity."
            if same else
            "One text uses a substantially broader or narrower range of unique words."
        )
    if "vocab_hapax" in name:
        return (
            "Both texts use rare, unique words at similar rates."
            if same else
            "One text uses far more rare, one-off words — a potential sophistication difference."
        )
    if "sent_mean" in name:
        return (
            "Average sentence length is similar across both texts."
            if same else
            "Average sentence length differs noticeably, suggesting different writing rhythms."
        )
    if "sent_std" in name or "sent_entropy" in name:
        return (
            "Sentence length variability is comparable in both texts."
            if same else
            "One text has much more variation in sentence length than the other."
        )
    if "passive_voice" in name:
        return (
            "Both texts use passive voice constructions at similar rates."
            if same else
            "Passive voice usage differs — one text favours active, the other passive constructions."
        )
    if "contraction_per_100" in name:
        return (
            "Both texts use contractions (e.g. don't, can't) at similar rates."
            if same else
            "One text uses contractions much more freely, suggesting a different formality level."
        )
    if "pronoun_first_person" in name:
        return (
            "Both texts use first-person pronouns (I, me, my) at similar rates."
            if same else
            "First-person pronoun usage differs notably between the two texts."
        )
    if "hedging" in name:
        return (
            "Both texts use hedging language (perhaps, might, seems) at similar rates."
            if same else
            "One text hedges claims much more frequently — a difference in writing style or certainty."
        )
    if "dep_depth" in name:
        return (
            "Grammatical sentence complexity is similar in both texts."
            if same else
            "One text uses noticeably more complex grammatical structures."
        )
    if "readability" in name or "fk_grade" in name or "gunning" in name:
        return (
            "Both texts sit at a similar reading difficulty level."
            if same else
            "The texts differ in reading difficulty — this may reflect topic or authorship."
        )
    if "discourse" in name:
        return (
            "Both texts use connecting words (therefore, however, because) at similar rates."
            if same else
            "One text relies more heavily on logical connectors in its argumentation."
        )
    if "punct_comma" in name:
        return (
            "Comma usage is similar, reflecting consistent punctuation habits."
            if same else
            "Comma frequency differs — a subtle but telling punctuation habit."
        )
    if "adv_sentence_initial" in name:
        return (
            "Both texts open sentences with adverbs at similar rates."
            if same else
            "One text opens sentences with adverbs far more often, reflecting a different style."
        )
    if "sent_opening" in name:
        return (
            "Sentence opening patterns (conjunctions, adverbs, pronouns) are similar."
            if same else
            "The texts differ in how sentences begin — a consistent stylistic habit."
        )
    return ""


def _build_educator_summary(debate: DebateResult) -> str:
    """Section 3: Judge's educator summary, verbatim."""
    summary = (debate.judge.educator_summary or "").strip()
    if not summary:
        return ""
    lines = [
        "Plain English Summary:",
        _DIVIDER,
    ]
    # Wrap each sentence-level chunk
    for paragraph in summary.split("\n"):
        para = paragraph.strip()
        if para:
            lines.append(_wrap(f'"{para}"', indent=_INDENT))
    lines.append("")
    return "\n".join(lines)


def _build_debate_record(debate: DebateResult) -> str:
    """Section 4: Decisive factors from the Judge's assessment."""
    factors = debate.judge.decisive_factors or []
    if not factors:
        return ""
    lines = [
        "Decisive Evidence (from the adjudication panel):",
        _DIVIDER,
    ]
    for i, factor in enumerate(factors, 1):
        wrapped = textwrap.fill(
            factor.strip(),
            width=_LINE_WIDTH,
            initial_indent=f"  {i}. ",
            subsequent_indent="     ",
        )
        lines.append(wrapped)
    lines.append("")

    # Analyst's key features cited
    analyst_features = debate.analyst.key_features_cited
    if analyst_features:
        lines.append("  Analyst cited features:")
        lines.append(f"    {', '.join(analyst_features)}")
        lines.append("")

    # Skeptic stance
    skeptic_stance = debate.skeptic.stance or ""
    if skeptic_stance:
        lines.append(f"  Skeptic stance: {skeptic_stance}")
        lines.append("")

    return "\n".join(lines)


def _build_consensus(debate: DebateResult) -> str:
    """Section 5: Agent consensus."""
    agreement_phrase = _agreement_label(debate.judge.agent_agreement)
    analyst_verdict  = _verdict_label(debate.analyst.verdict)
    judge_verdict    = _verdict_label(debate.final_verdict)

    lines = [
        "Agent Consensus:",
        _DIVIDER,
        f"  Analyst:  {analyst_verdict} "
        f"({round(debate.analyst.confidence * 100)}% confidence)",
        f"  Skeptic:  {debate.skeptic.stance or 'N/A'}",
        f"  Judge:    {judge_verdict} "
        f"({round(debate.final_confidence * 100)}% confidence)",
        f"  {agreement_phrase}",
        "",
    ]
    return "\n".join(lines)


def _build_disclaimer() -> str:
    """Section 6: Mandatory disclaimer."""
    lines = [
        _DIVIDER,
        _wrap(
            "IMPORTANT: This is a flagging tool for educator review, not a final "
            "judgment. Stylometric analysis is probabilistic and can be affected "
            "by topic differences, text length, and writing context. All flagged "
            "cases should be reviewed by a qualified human before any action is taken.",
            indent=_INDENT,
        ),
        _BORDER,
    ]
    return "\n".join(lines)


def _build_metadata(debate: DebateResult) -> str:
    """Footer: timestamp and processing time."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    elapsed = f"{debate.elapsed_seconds:.1f}s" if debate.elapsed_seconds else "N/A"
    return f"\nGenerated: {ts}  |  Debate processing time: {elapsed}\n"


# ──────────────────────────────── Main Formatter ──────────────────────────────

def format_report(
    debate: DebateResult,
    shap_result=None,   # Optional[ShapResult] from shap_analysis.py
    include_metadata: bool = True,
) -> str:
    """Format a complete educator-facing authorship verification report.

    Parameters
    ----------
    debate           : Completed DebateResult from ``DebateOrchestrator.run()``.
    shap_result      : Optional ShapResult from ``explain_pair()``.  When provided,
                       the SHAP signals section is included; when None, it is omitted
                       gracefully.
    include_metadata : Whether to append timestamp and elapsed-time footer.

    Returns
    -------
    Formatted plain-text report string, ready to print or save.
    """
    sections = [
        _build_header(debate),
        _build_shap_signals(shap_result),
        _build_educator_summary(debate),
        _build_debate_record(debate),
        _build_consensus(debate),
        _build_disclaimer(),
    ]
    if include_metadata:
        sections.append(_build_metadata(debate))

    # Filter out empty sections and join
    return "\n".join(s for s in sections if s.strip())


# ──────────────────────────────── Persistence ─────────────────────────────────

def save_report(
    report: str,
    path: str,
) -> str:
    """Write a formatted report string to a plain-text file.

    Parameters
    ----------
    report : Output of ``format_report()``.
    path   : Destination file path.

    Returns
    -------
    Absolute path of the written file.
    """
    fpath = Path(path)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(report, encoding="utf-8")
    return str(fpath.resolve())


def format_and_save(
    debate: DebateResult,
    shap_result=None,   # Optional[ShapResult]
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
    pair_id: Optional[str] = None,
) -> str:
    """Format a report and save it to ``output_dir``.

    Parameters
    ----------
    debate     : Completed DebateResult.
    shap_result: Optional ShapResult.
    output_dir : Directory to write into.  Defaults to ``experiments/results/``.
    filename   : Override the auto-generated filename.
    pair_id    : Optional pair identifier appended to the filename.

    Returns
    -------
    Path to the saved file.
    """
    report = format_report(debate, shap_result=shap_result)

    out_dir = Path(output_dir or "experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename:
        fpath = out_dir / filename
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = f"_{pair_id}" if pair_id else ""
        verdict_tag = debate.final_verdict.lower()
        fpath = out_dir / f"av_report{suffix}_{verdict_tag}_{ts}.txt"

    saved_path = save_report(report, str(fpath))
    print(f"[trace_parser] Report saved → {saved_path}")
    return saved_path


# ──────────────────────────────── Batch Formatter ────────────────────────────

def format_batch(
    debates: List[DebateResult],
    shap_results: Optional[List] = None,    # Optional[List[ShapResult]]
    output_dir: Optional[str] = None,
    pair_ids: Optional[List[str]] = None,
) -> List[str]:
    """Format and save reports for a batch of debates.

    Parameters
    ----------
    debates      : List of DebateResult objects.
    shap_results : Corresponding ShapResult list (or None to omit SHAP sections).
    output_dir   : Where to save reports.
    pair_ids     : Optional list of identifiers for file naming.

    Returns
    -------
    List of saved file paths, one per debate.
    """
    paths: List[str] = []
    n = len(debates)
    for i, debate in enumerate(debates):
        shap = shap_results[i] if shap_results and i < len(shap_results) else None
        pid  = pair_ids[i] if pair_ids and i < len(pair_ids) else str(i)
        path = format_and_save(debate, shap_result=shap,
                               output_dir=output_dir, pair_id=pid)
        paths.append(path)
        if (i + 1) % 10 == 0:
            print(f"[trace_parser] {i+1}/{n} reports saved …")
    print(f"[trace_parser] Batch complete. {n} reports saved to {output_dir or 'experiments/results/'}")
    return paths
