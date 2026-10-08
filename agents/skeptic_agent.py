"""Skeptic/Challenger agent (Agent B in the debate framework).

Receives Agent A's verdict and evidence packet, then produces a structured
challenge identifying topic confounds, overlooked low-delta features, and
alternative interpretations.

Public API
----------
    skeptic = SkepticAgent()
    response = skeptic.challenge(pair, evidence_packet, analyst_response)
    # response is a SkepticResponse
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import config
from agents.base_agent import AgentResponse, BaseAgent
from data.preprocessor import TextPair

_PROMPT_PATH = Path(__file__).parent / "prompts" / "skeptic_prompt.txt"
_TEXT_PREVIEW_CHARS = 2500


# ──────────────────────────────── Data Contract ───────────────────────────────

@dataclass
class SkepticResponse:
    """Structured output of the Skeptic agent.

    Fields
    ------
    stance              : "AGREE", "DISAGREE", or "PARTIALLY_DISAGREE".
    confidence          : Float in [0.0, 1.0] — the Skeptic's confidence in its
                          own stance.
    challenges          : Specific critiques of Agent A's reasoning.
    overlooked_evidence : Feature deltas Agent A underweighted.
    revised_reasoning   : Skeptic's own 2–3 paragraph analysis.
    raw_response        : Complete unmodified LLM output.
    parse_ok            : False if any field failed to parse cleanly.
    """

    stance: str
    confidence: float
    challenges: List[str]
    overlooked_evidence: List[str]
    revised_reasoning: str
    raw_response: str
    parse_ok: bool = field(default=True)


# ──────────────────────────────── Response Parser ─────────────────────────────

_STANCE_RE = re.compile(
    r"STANCE\s*:\s*(AGREE|DISAGREE|PARTIALLY_DISAGREE)",
    re.IGNORECASE,
)
_CONFIDENCE_RE = re.compile(
    r"CONFIDENCE\s*:\s*\[?\s*([0-9]*\.?[0-9]+)\s*\]?",
)
# Match the bullet-list block between CHALLENGES: and the next section header
_CHALLENGES_RE = re.compile(
    r"CHALLENGES\s*:\s*(.*?)(?=\nOVERLOOKED_EVIDENCE\s*:|\nREVISED_REASONING\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)
_OVERLOOKED_RE = re.compile(
    r"OVERLOOKED_EVIDENCE\s*:\s*(.*?)(?=\nREVISED_REASONING\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)
_REVISED_RE = re.compile(
    r"REVISED_REASONING\s*:\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)


def _parse_bullet_list(raw_block: str) -> List[str]:
    """Extract dash-prefixed or plain lines from a multi-line block."""
    items = []
    for line in raw_block.strip().splitlines():
        line = line.strip().lstrip("-•* ").strip()
        if line:
            items.append(line)
    return items


def _parse_skeptic_response(raw: str) -> SkepticResponse:
    """Extract structured fields from the Skeptic's raw LLM output."""
    parse_ok = True

    sm = _STANCE_RE.search(raw)
    stance = sm.group(1).upper() if sm else "UNKNOWN"
    if not sm:
        parse_ok = False
        warnings.warn(
            "SkepticResponse parse: could not find STANCE field.",
            stacklevel=3,
        )

    cm = _CONFIDENCE_RE.search(raw)
    if cm:
        raw_conf = float(cm.group(1))
        confidence = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf
        confidence = max(0.0, min(1.0, confidence))
    else:
        confidence = 0.5
        parse_ok = False

    chm = _CHALLENGES_RE.search(raw)
    challenges = _parse_bullet_list(chm.group(1)) if chm else []
    if not chm:
        parse_ok = False

    om = _OVERLOOKED_RE.search(raw)
    overlooked_evidence = _parse_bullet_list(om.group(1)) if om else []
    if not om:
        parse_ok = False

    rm = _REVISED_RE.search(raw)
    revised_reasoning = rm.group(1).strip() if rm else raw.strip()
    if not rm:
        parse_ok = False

    return SkepticResponse(
        stance=stance,
        confidence=confidence,
        challenges=challenges,
        overlooked_evidence=overlooked_evidence,
        revised_reasoning=revised_reasoning,
        raw_response=raw,
        parse_ok=parse_ok,
    )


# ──────────────────────────────── Message Builder ─────────────────────────────

def _build_skeptic_message(
    pair: TextPair,
    evidence_packet: str,
    analyst_response: AgentResponse,
) -> str:
    """Assemble the user message for the Skeptic."""
    def _truncate(text: str) -> tuple[str, bool]:
        if len(text) <= _TEXT_PREVIEW_CHARS:
            return text, False
        return text[:_TEXT_PREVIEW_CHARS], True

    text_a, a_trunc = _truncate(pair.text_a)
    text_b, b_trunc = _truncate(pair.text_b)
    a_note = f"\n[...truncated to {_TEXT_PREVIEW_CHARS} chars]" if a_trunc else ""
    b_note = f"\n[...truncated to {_TEXT_PREVIEW_CHARS} chars]" if b_trunc else ""

    features_str = (
        "\n".join(f"  • {f}" for f in analyst_response.key_features_cited)
        if analyst_response.key_features_cited else "  (none listed)"
    )

    return "\n".join([
        "=== TEXT A (Reference Document) ===",
        f"{text_a}{a_note}",
        "",
        "=== TEXT B (Unknown Document to Verify) ===",
        f"{text_b}{b_note}",
        "",
        evidence_packet,
        "",
        "=== AGENT A (ANALYST) VERDICT AND REASONING ===",
        f"Verdict    : {analyst_response.verdict}",
        f"Confidence : {analyst_response.confidence:.2f}",
        f"Key features cited:\n{features_str}",
        "",
        "Reasoning:",
        analyst_response.reasoning,
    ])


# ──────────────────────────────── SkepticAgent ────────────────────────────────

class SkepticAgent(BaseAgent):
    """Agent B — Skeptic/Challenger.

    Challenges Agent A's verdict by identifying topic confounds, overlooked
    LOW-delta features, and alternative explanations.

    Parameters
    ----------
    model       : LLM model name. Defaults to ``config.SKEPTIC_MODEL``.
    temperature : Sampling temperature. Defaults to ``config.AGENT_TEMPERATURE``.
    prompt_path : Override prompt file location (useful in tests).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        prompt_path: Optional[Path] = None,
    ) -> None:
        path = prompt_path or _PROMPT_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Skeptic prompt file not found: {path}\n"
                "Expected location: agents/prompts/skeptic_prompt.txt"
            )
        super().__init__(
            model=model or config.SKEPTIC_MODEL,
            system_prompt=path.read_text(encoding="utf-8").strip(),
            temperature=temperature if temperature is not None else config.AGENT_TEMPERATURE,
        )

    def challenge(
        self,
        pair: TextPair,
        evidence_packet: str,
        analyst_response: AgentResponse,
    ) -> SkepticResponse:
        """Challenge Agent A's verdict.

        Parameters
        ----------
        pair             : The text pair being analysed.
        evidence_packet  : Output of ``format_evidence_packet``.
        analyst_response : Agent A's ``AgentResponse``.

        Returns
        -------
        ``SkepticResponse`` with stance, challenges, overlooked evidence,
        and revised reasoning.
        """
        user_message = _build_skeptic_message(pair, evidence_packet, analyst_response)
        raw = self._raw_call(user_message)
        return _parse_skeptic_response(raw)
