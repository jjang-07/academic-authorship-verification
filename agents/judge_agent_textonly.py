"""Text-only Judge agent — final verdict from close reading and debate transcript.

Differences from JudgeAgent
-----------------------------
- The user message contains only TEXT A, TEXT B, and the full debate transcript
  (Analyst verdict + Skeptic challenge + optional rebuttal).
  No evidence packet, no numeric deltas, no severity labels.
- The system prompt instructs the Judge to re-read both texts independently,
  then evaluate each agent's argument against what the texts actually show.
- Returns the identical ``JudgeResponse`` dataclass so all downstream code is
  unchanged.

Public API
----------
    judge    = JudgeAgentTextOnly()
    response = judge.adjudicate(pair, analyst, skeptic, rebuttal=None)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import config
from agents.base_agent import AgentResponse, BaseAgent
from agents.judge_agent import JudgeResponse, _parse_judge_response
from agents.skeptic_agent import SkepticResponse
from data.preprocessor import TextPair

_TEXTONLY_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge_prompt_textonly.txt"
_TEXT_PREVIEW_CHARS = 2500


def _load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Text-only Judge prompt not found: {path}\n"
            "Expected location: agents/prompts/judge_prompt_textonly.txt"
        )
    return path.read_text(encoding="utf-8").strip()


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= _TEXT_PREVIEW_CHARS:
        return text, False
    return text[:_TEXT_PREVIEW_CHARS], True


def _bullet_list(items: List[str]) -> str:
    return "\n".join(f"  - {i}" for i in items) if items else "  (none)"


def _features_str(features: List[str]) -> str:
    return (
        "\n".join(f"  • {f}" for f in features)
        if features else "  (none listed)"
    )


def _build_message(
    pair: TextPair,
    analyst_response: AgentResponse,
    skeptic_response: SkepticResponse,
    rebuttal: Optional[AgentResponse] = None,
) -> str:
    """User message: raw texts + full debate transcript — no evidence packet."""
    text_a, a_trunc = _truncate(pair.text_a)
    text_b, b_trunc = _truncate(pair.text_b)
    a_note = f"\n[...truncated to {_TEXT_PREVIEW_CHARS} chars]" if a_trunc else ""
    b_note = f"\n[...truncated to {_TEXT_PREVIEW_CHARS} chars]" if b_trunc else ""

    parts = [
        "TEXT A:",
        f"{text_a}{a_note}",
        "",
        "TEXT B:",
        f"{text_b}{b_note}",
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


class JudgeAgentTextOnly(BaseAgent):
    """Text-only variant of Agent C (Judge).

    Re-reads both texts and evaluates the debate transcript without access to
    any numeric feature measurements. Produces the same ``JudgeResponse``
    as the standard JudgeAgent.

    Parameters
    ----------
    model       : LLM model name.  Defaults to ``config.JUDGE_MODEL``.
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
            model=model or config.JUDGE_MODEL,
            system_prompt=sp,
            temperature=temperature if temperature is not None else config.AGENT_TEMPERATURE,
        )

    def adjudicate(
        self,
        pair: TextPair,
        analyst_response: AgentResponse,
        skeptic_response: SkepticResponse,
        rebuttal: Optional[AgentResponse] = None,
    ) -> JudgeResponse:
        """Produce a final verdict after the full text-only debate.

        Parameters
        ----------
        pair             : The text pair being analysed.
        analyst_response : Agent A's ``AgentResponse``.
        skeptic_response : Agent B's ``SkepticResponse``.
        rebuttal         : Optional Agent A round-2 rebuttal (``AgentResponse``).

        Returns
        -------
        ``JudgeResponse`` — same dataclass as the standard JudgeAgent.
        """
        raw = self._raw_call(_build_message(
            pair, analyst_response, skeptic_response, rebuttal
        ))
        return _parse_judge_response(raw)
