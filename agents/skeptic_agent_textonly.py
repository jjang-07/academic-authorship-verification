"""Text-only Skeptic agent — challenges the Analyst from close reading alone.

Differences from SkepticAgent
------------------------------
- The user message contains only TEXT A, TEXT B, and the Analyst's verdict.
  No evidence packet, no numeric deltas, no severity labels.
- The system prompt instructs the Skeptic to re-read both texts and challenge
  the Analyst by finding micro-habits that were overlooked or misattributed to
  topic differences rather than authorship.
- Returns the identical ``SkepticResponse`` dataclass so all downstream code
  (JudgeAgent, DebateOrchestrator, test_debate.py) is unchanged.

Public API
----------
    skeptic  = SkepticAgentTextOnly()
    response = skeptic.challenge(pair, analyst_response)  # no evidence_packet
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import config
from agents.base_agent import AgentResponse, BaseAgent
from agents.skeptic_agent import SkepticResponse, _parse_skeptic_response
from data.preprocessor import TextPair

_TEXTONLY_PROMPT_PATH = Path(__file__).parent / "prompts" / "skeptic_prompt_textonly.txt"
_TEXT_PREVIEW_CHARS = 2500


def _load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Text-only Skeptic prompt not found: {path}\n"
            "Expected location: agents/prompts/skeptic_prompt_textonly.txt"
        )
    return path.read_text(encoding="utf-8").strip()


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= _TEXT_PREVIEW_CHARS:
        return text, False
    return text[:_TEXT_PREVIEW_CHARS], True


def _build_message(pair: TextPair, analyst_response: AgentResponse) -> str:
    """User message: raw texts + Analyst verdict only — no evidence packet."""
    text_a, a_trunc = _truncate(pair.text_a)
    text_b, b_trunc = _truncate(pair.text_b)
    a_note = f"\n[...truncated to {_TEXT_PREVIEW_CHARS} chars]" if a_trunc else ""
    b_note = f"\n[...truncated to {_TEXT_PREVIEW_CHARS} chars]" if b_trunc else ""

    features_str = (
        "\n".join(f"  • {f}" for f in analyst_response.key_features_cited)
        if analyst_response.key_features_cited else "  (none listed)"
    )

    return "\n".join([
        "TEXT A:",
        f"{text_a}{a_note}",
        "",
        "TEXT B:",
        f"{text_b}{b_note}",
        "",
        "=== AGENT A (ANALYST) VERDICT AND REASONING ===",
        f"Verdict    : {analyst_response.verdict}",
        f"Confidence : {analyst_response.confidence:.2f}",
        f"Key features cited:\n{features_str}",
        "",
        "Reasoning:",
        analyst_response.reasoning,
    ])


class SkepticAgentTextOnly(BaseAgent):
    """Text-only variant of Agent B (Skeptic).

    Re-reads both texts independently to challenge the Analyst's close-reading
    verdict, without access to any numeric feature measurements.

    Parameters
    ----------
    model       : LLM model name.  Defaults to ``config.SKEPTIC_MODEL``.
    temperature : Sampling temperature.  Defaults to ``config.AGENT_TEMPERATURE``.
    prompt_path : Override prompt file path (useful in tests).
    system_prompt : If set, use this system text instead of loading ``prompt_path``.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        prompt_path: Optional[Path] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        if system_prompt is not None:
            sp = system_prompt.strip()
        else:
            sp = _load_prompt(prompt_path or _TEXTONLY_PROMPT_PATH)
        super().__init__(
            model=model or config.SKEPTIC_MODEL,
            system_prompt=sp,
            temperature=temperature if temperature is not None else config.AGENT_TEMPERATURE,
        )

    def challenge(
        self,
        pair: TextPair,
        analyst_response: AgentResponse,
    ) -> SkepticResponse:
        """Challenge the Analyst's verdict from close reading alone.

        Parameters
        ----------
        pair             : The text pair being analysed.
        analyst_response : Agent A's ``AgentResponse``.

        Returns
        -------
        ``SkepticResponse`` — same dataclass as the standard SkepticAgent.
        """
        raw = self._raw_call(_build_message(pair, analyst_response))
        return _parse_skeptic_response(raw)
