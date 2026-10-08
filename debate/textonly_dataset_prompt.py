"""Dataset-specific injections for text-only debate system prompts.

When ``dataset_context='student_essays'``, the wire-service warning sections in
the Analyst, Skeptic, and Judge text-only prompts are replaced with an
ACADEMIC ESSAY WARNING. Applied by ``DebateOrchestratorTextOnly`` on init.
"""

from __future__ import annotations

import re
import warnings
from typing import Literal, Optional

# Value passed from ``run_student_experiment`` / ``test_debate --dataset student``
DATASET_CONTEXT_STUDENT_ESSAYS = "student_essays"

# Human-readable label for default text-only prompts (wire-service warning blocks; no injection).
DATASET_CONTEXT_WIRE_SERVICE = "wire_service"

Role = Literal["analyst", "skeptic", "judge"]

# Shared body (no title line). Analyst uses a rule line under the title like the wire block.
_STUDENT_WARNING_BODY = """These texts are student academic essays, not wire-service journalism. Apply these dataset-specific guidelines:
(1) ESSAY TYPE CONFOUND — a persuasive essay and a personal narrative by the same author will differ in structure, transition style, and register by design. Do not treat structural differences between essay types as authorship evidence. Focus only on habits that persist across essay types.
(2) TOPIC VOCABULARY CONFOUND — essays on different subjects will use completely different domain vocabulary. Discount all topic-specific terminology exactly as you would domain words in news articles.
(3) ASSIGNMENT CONSTRAINTS — discount these as assignment-driven: formal transition markers (furthermore, additionally, moreover, next), essay organizational structure (thesis statements, topic sentences, conclusion paragraphs), quote integration format (ICE/quote sandwich method), and sentence length differences between narrative and analytical essays.
Do NOT discount these as assignment-driven — they are stable individual habits:

Contraction use: a student who habitually uses contractions will tend to use them even in formal writing unless explicitly prohibited. Complete absence vs. heavy use across texts is a strong authorship signal.
Systematic capitalization patterns: consistent sentence-initial lowercasing throughout a text is an idiosyncratic habit, not an assignment requirement. Verify carefully whether each text has a consistent internal pattern before treating capitalization as shared or divergent evidence. CRITICAL — capitalization asymmetry is a strong DIFFERENT_AUTHOR signal. If one text is systematically lowercase throughout including proper nouns and sentence-initial positions, while the other text capitalizes normally, this is a significant divergence that survives the essay-type check. Do not dismiss this as assignment-driven — verify carefully whether each text has a consistent internal capitalization pattern, and treat asymmetric patterns as meaningful authorship evidence.
Mechanical error patterns: apostrophe misuse, comma splice tendencies, and run-on sentence habits are individual and persist across assignment types.

Apply judgment for register and rhetorical questions — these are partially assignment-driven. A persuasive essay naturally invites more colloquial register and rhetorical questions than a literary analysis. However, if a student uses rhetorical questions habitually across multiple essay types, or if the register difference is extreme, these can still carry modest authorship weight.
(4) ACADEMIC GENRE CONVENTIONS — other widely shared scaffolding (evidence-claim-warrant patterns, citation framing boilerplate, formulaic peer-review or academic phrasing not already listed in (3)) is genre-wide — discount it the same way wire-service conventions are discounted unless it clearly varies between individual writers.
What to look for instead: Focus on writing-style micro-habits that would persist if the same author wrote both essays on the same topic and essay type — sentence rhythm, characteristic syntax, punctuation, and transition preferences; plus the stable individual signals above (contractions, internal capitalization consistency, mechanical error patterns) when verified in-text. Use nuanced judgment for register and rhetorical questions per (3). If a habit is explained by essay type or assignment, discard it."""

# Judge-only: inserted after capitalization asymmetry guidance, before mechanical errors.
_JUDGE_MECHANICAL_MARKER = "Mechanical error patterns:"
_STUDENT_JUDGE_FLIP_THRESHOLD = (
    "FLIPPING THRESHOLD FOR STUDENT ESSAYS: The general three-condition flip rule still "
    "applies, but when the Skeptic's additive counter-evidence includes any of the stable "
    "individual signals (contraction asymmetry, systematic capitalization asymmetry, or "
    "distinctive mechanical error patterns), these alone are sufficient to meet condition 2 "
    "of the flip threshold — they are strong enough individual signals that they do not need "
    "to be combined with full invalidation of the Analyst's primary evidence. If the Skeptic "
    "identifies a clear capitalization asymmetry, contraction asymmetry, or distinctive error "
    "pattern that you can verify in the texts, treat this as strong additive counter-evidence "
    "that justifies reducing confidence significantly or flipping, even if the Analyst's "
    "primary evidence partially survives the topic-check."
)

_judge_body_parts = _STUDENT_WARNING_BODY.split(_JUDGE_MECHANICAL_MARKER, 1)
if len(_judge_body_parts) != 2:
    raise RuntimeError(
        "textonly_dataset_prompt: _STUDENT_WARNING_BODY must contain exactly one "
        f"{_JUDGE_MECHANICAL_MARKER!r} for judge-only flip-threshold insertion."
    )
_STUDENT_JUDGE_WARNING_BODY = (
    _judge_body_parts[0].rstrip()
    + "\n\n"
    + _STUDENT_JUDGE_FLIP_THRESHOLD
    + "\n\n"
    + _JUDGE_MECHANICAL_MARKER
    + _judge_body_parts[1]
)

_STUDENT_ANALYST_BLOCK = (
    "ACADEMIC ESSAY WARNING\n"
    "────────────────────\n"
    + _STUDENT_WARNING_BODY
    + "\n"
)

# Skeptic wire block uses two-space indentation on continuation lines.
_STUDENT_SKEPTIC_BLOCK = (
    "ACADEMIC ESSAY WARNING\n"
    + "\n".join(
        ("  " + ln) if ln.strip() else ""
        for ln in _STUDENT_WARNING_BODY.splitlines()
    )
    + "\n"
)

# Judge STEP 1 wire paragraph: two-space indent per line (title + body + judge-only flip rule).
_STUDENT_JUDGE_BLOCK = (
    "  ACADEMIC ESSAY WARNING\n"
    + "\n".join(
        "  " + ln if ln.strip() else "" for ln in _STUDENT_JUDGE_WARNING_BODY.splitlines()
    )
    + "\n"
)

# Analyst: from WIRE-SERVICE title through end of wire section (before HOW TO REASON).
_ANALYST_WIRE_RE = re.compile(
    r"WIRE-SERVICE WARNING\n────────────────────\n.*?\na committed verdict\.\n\n"
    r"(?=\n*═+\nHOW TO REASON TOWARD A VERDICT)",
    re.DOTALL,
)

# Skeptic: wire block before PRE-WRITING SELF-CHECK.
_SKEPTIC_WIRE_RE = re.compile(
    r"WIRE-SERVICE WARNING\n  If both texts are wire-service.*?\n  author\.\n\n"
    r"(?=PRE-WRITING SELF-CHECK)",
    re.DOTALL,
)

# Judge: wire paragraph inside STEP 1 (before STEP 2).
_JUDGE_WIRE_RE = re.compile(
    r"  If both texts appear to be wire-service news articles.*?\n  between writers\.\n\n"
    r"(?=STEP 2)",
    re.DOTALL,
)


def apply_textonly_dataset_context(
    system_prompt: str,
    role: Role,
    dataset_context: Optional[str],
) -> str:
    """Return ``system_prompt`` with dataset-specific section applied when needed.

    For ``student_essays``, replaces the wire-service warning region per role.
    Unknown ``dataset_context`` values raise ``ValueError``.
    """
    if dataset_context is None:
        return system_prompt
    if dataset_context != DATASET_CONTEXT_STUDENT_ESSAYS:
        raise ValueError(
            f"Unknown dataset_context={dataset_context!r}; "
            f"expected None or {DATASET_CONTEXT_STUDENT_ESSAYS!r}."
        )

    if role == "analyst":
        new_prompt, n = _ANALYST_WIRE_RE.subn(_STUDENT_ANALYST_BLOCK, system_prompt, count=1)
        if n == 0:
            warnings.warn(
                "student_essays: analyst WIRE-SERVICE block not found; prompt unchanged",
                stacklevel=2,
            )
            return system_prompt
        return new_prompt

    if role == "skeptic":
        new_prompt, n = _SKEPTIC_WIRE_RE.subn(_STUDENT_SKEPTIC_BLOCK, system_prompt, count=1)
        if n == 0:
            warnings.warn(
                "student_essays: skeptic WIRE-SERVICE block not found; prompt unchanged",
                stacklevel=2,
            )
            return system_prompt
        new_prompt = new_prompt.replace(
            "Wire-service check: apply the WIRE-SERVICE WARNING above.",
            "Essay-context check: apply the ACADEMIC ESSAY WARNING above.",
        )
        return new_prompt

    if role == "judge":
        new_prompt, n = _JUDGE_WIRE_RE.subn(_STUDENT_JUDGE_BLOCK, system_prompt, count=1)
        if n == 0:
            warnings.warn(
                "student_essays: judge wire-service paragraph in STEP 1 not found; prompt unchanged",
                stacklevel=2,
            )
            return system_prompt
        return new_prompt

    raise ValueError(f"Unknown role: {role!r}")
