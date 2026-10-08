"""Judge/Integrator agent (Agent C in the debate framework).

Receives the full debate record — evidence packet, Analyst verdict, Skeptic
challenge, and optional Analyst rebuttal — then produces a final calibrated
authorship verdict with an educator-facing plain-English summary.

Public API
----------
    judge = JudgeAgent()
    response = judge.adjudicate(pair, evidence_packet,
                                analyst_response, skeptic_response,
                                rebuttal=None)
    # response is a JudgeResponse
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import config
from agents.base_agent import AgentResponse, BaseAgent
from agents.skeptic_agent import SkepticResponse
from data.preprocessor import TextPair

_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge_prompt.txt"
_TEXT_PREVIEW_CHARS = 2500


# ──────────────────────────────── Data Contract ───────────────────────────────

@dataclass
class JudgeResponse:
    """Structured output of the Judge agent.

    Fields
    ------
    verdict           : "SAME_AUTHOR" or "DIFFERENT_AUTHOR".
    confidence        : Float in [0.0, 1.0].
    decisive_factors  : Top 3 features/arguments that determined the verdict.
    educator_summary  : Plain-English paragraph for non-expert readers.
    agent_agreement   : "FULL", "PARTIAL", or "NONE".
    raw_response      : Complete unmodified LLM output.
    parse_ok          : False if any field failed to parse cleanly.
    """

    verdict: str
    confidence: float
    decisive_factors: List[str]
    educator_summary: str
    agent_agreement: str
    raw_response: str
    parse_ok: bool = field(default=True)


# ──────────────────────────────── Response Parser ─────────────────────────────

_VERDICT_RE = re.compile(
    r"FINAL_VERDICT\s*:\s*\[?\s*(SAME_AUTHOR|DIFFERENT_AUTHOR)\s*\]?",
    re.IGNORECASE,
)
_CONFIDENCE_RE = re.compile(
    r"FINAL_CONFIDENCE\s*:\s*\[?\s*([0-9]*\.?[0-9]+)\s*\]?",
)
_FACTORS_RE = re.compile(
    r"DECISIVE_FACTORS\s*:\s*(.*?)(?=\nEDUCATOR_SUMMARY\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)
_SUMMARY_RE = re.compile(
    r"EDUCATOR_SUMMARY\s*:\s*(.*?)(?=\nAGENT_AGREEMENT\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)
_AGREEMENT_RE = re.compile(
    r"AGENT_AGREEMENT\s*:\s*(FULL|PARTIAL|NONE)",
    re.IGNORECASE,
)


def _parse_bullet_list(raw_block: str) -> List[str]:
    items = []
    for line in raw_block.strip().splitlines():
        line = line.strip().lstrip("-•* ").strip()
        if line:
            items.append(line)
    return items


def _parse_judge_response(raw: str) -> JudgeResponse:
    """Extract structured fields from the Judge's raw LLM output."""
    parse_ok = True

    vm = _VERDICT_RE.search(raw)
    verdict = vm.group(1).upper() if vm else "UNKNOWN"
    if not vm:
        parse_ok = False
        warnings.warn(
            "JudgeResponse parse: could not find FINAL_VERDICT field.",
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

    fm = _FACTORS_RE.search(raw)
    decisive_factors = _parse_bullet_list(fm.group(1)) if fm else []
    if not fm:
        parse_ok = False

    sm = _SUMMARY_RE.search(raw)
    educator_summary = sm.group(1).strip() if sm else ""
    if not sm:
        parse_ok = False

    am = _AGREEMENT_RE.search(raw)
    agent_agreement = am.group(1).upper() if am else "UNKNOWN"
    if not am:
        parse_ok = False

    return JudgeResponse(
        verdict=verdict,
        confidence=confidence,
        decisive_factors=decisive_factors,
        educator_summary=educator_summary,
        agent_agreement=agent_agreement,
        raw_response=raw,
        parse_ok=parse_ok,
    )


# ──────────────────────────────── Message Builder ─────────────────────────────

def _build_judge_message(
    pair: TextPair,
    evidence_packet: str,
    analyst_response: AgentResponse,
    skeptic_response: SkepticResponse,
    rebuttal: Optional[AgentResponse] = None,
) -> str:
    """Assemble the full debate transcript for the Judge."""
    def _truncate(text: str) -> tuple[str, bool]:
        if len(text) <= _TEXT_PREVIEW_CHARS:
            return text, False
        return text[:_TEXT_PREVIEW_CHARS], True

    text_a, a_trunc = _truncate(pair.text_a)
    text_b, b_trunc = _truncate(pair.text_b)
    a_note = f"\n[...truncated to {_TEXT_PREVIEW_CHARS} chars]" if a_trunc else ""
    b_note = f"\n[...truncated to {_TEXT_PREVIEW_CHARS} chars]" if b_trunc else ""

    def _features_str(features: List[str]) -> str:
        return (
            "\n".join(f"  • {f}" for f in features)
            if features else "  (none listed)"
        )

    def _bullet_list(items: List[str]) -> str:
        return "\n".join(f"  - {i}" for i in items) if items else "  (none)"

    parts = [
        "=== TEXT A (Reference Document) ===",
        f"{text_a}{a_note}",
        "",
        "=== TEXT B (Unknown Document to Verify) ===",
        f"{text_b}{b_note}",
        "",
        evidence_packet,
        "",
        "=== AGENT A (ANALYST) — INITIAL VERDICT ===",
        f"Verdict    : {analyst_response.verdict}",
        f"Confidence : {analyst_response.confidence:.2f}",
        f"Key features cited:\n{_features_str(analyst_response.key_features_cited)}",
        "",
        "Reasoning:",
        analyst_response.reasoning,
        "",
        "=== AGENT B (SKEPTIC) — CHALLENGE ===",
        f"Stance     : {skeptic_response.stance}",
        f"Confidence : {skeptic_response.confidence:.2f}",
        "",
        "Challenges:",
        _bullet_list(skeptic_response.challenges),
        "",
        "Overlooked evidence:",
        _bullet_list(skeptic_response.overlooked_evidence),
        "",
        "Revised reasoning:",
        skeptic_response.revised_reasoning,
    ]

    if rebuttal is not None:
        parts += [
            "",
            "=== AGENT A REBUTTAL (Round 2) ===",
            f"Updated verdict    : {rebuttal.verdict}",
            f"Updated confidence : {rebuttal.confidence:.2f}",
            "",
            "Rebuttal reasoning:",
            rebuttal.reasoning,
        ]

    return "\n".join(parts)


# ──────────────────────────────── JudgeAgent ──────────────────────────────────

class JudgeAgent(BaseAgent):
    """Agent C — Judge/Integrator.

    Weighs the Analyst's verdict and the Skeptic's challenge to produce a
    final calibrated verdict with an educator-facing plain-English summary.

    Parameters
    ----------
    model       : LLM model name. Defaults to ``config.JUDGE_MODEL``.
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
                f"Judge prompt file not found: {path}\n"
                "Expected location: agents/prompts/judge_prompt.txt"
            )
        super().__init__(
            model=model or config.JUDGE_MODEL,
            system_prompt=path.read_text(encoding="utf-8").strip(),
            temperature=temperature if temperature is not None else config.AGENT_TEMPERATURE,
        )

    def adjudicate(
        self,
        pair: TextPair,
        evidence_packet: str,
        analyst_response: AgentResponse,
        skeptic_response: SkepticResponse,
        rebuttal: Optional[AgentResponse] = None,
    ) -> JudgeResponse:
        """Produce a final calibrated verdict after the full debate.

        Parameters
        ----------
        pair             : The text pair being analysed.
        evidence_packet  : Output of ``format_evidence_packet``.
        analyst_response : Agent A's initial ``AgentResponse``.
        skeptic_response : Agent B's ``SkepticResponse``.
        rebuttal         : Optional Agent A round-2 rebuttal ``AgentResponse``.

        Returns
        -------
        ``JudgeResponse`` with final verdict, confidence, decisive factors,
        educator summary, and agent agreement label.
        """
        user_message = _build_judge_message(
            pair, evidence_packet, analyst_response, skeptic_response, rebuttal
        )
        raw = self._raw_call(user_message)
        return _parse_judge_response(raw)
