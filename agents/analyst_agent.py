"""Stylometric Analyst agent (Agent A in the debate framework).

Responsibilities
----------------
- Load the analyst system prompt from ``agents/prompts/analyst_prompt.txt``.
- Accept a ``TextPair`` and a pre-formatted evidence packet string.
- Build the user message that presents both texts plus the evidence packet.
- Delegate the API call and response parsing to ``BaseAgent``.

Public API
----------
    agent = AnalystAgent()                    # uses ANALYST_MODEL from config
    response = agent.analyze(pair, packet)    # returns AgentResponse
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import config
from agents.base_agent import AgentResponse, BaseAgent
from data.preprocessor import TextPair

# Path to the analyst system prompt relative to this file
_PROMPT_PATH = Path(__file__).parent / "prompts" / "analyst_prompt.txt"

# Characters to show per text in the user message.
# Keeps token count bounded during development; increase for production runs.
_TEXT_PREVIEW_CHARS = 2500


def _load_prompt(path: Path = _PROMPT_PATH) -> str:
    """Read the analyst system prompt from disk.

    Raises
    ------
    FileNotFoundError
        If the prompt file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Analyst prompt file not found: {path}\n"
            "Expected location: agents/prompts/analyst_prompt.txt"
        )
    return path.read_text(encoding="utf-8").strip()


def _truncate(text: str, max_chars: int = _TEXT_PREVIEW_CHARS) -> tuple[str, bool]:
    """Return (possibly truncated text, was_truncated)."""
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _build_user_message(
    pair: TextPair,
    evidence_packet: str,
    retrieved_context: Optional[str] = None,
) -> str:
    """Assemble the full user message sent to the analyst.

    Structure
    ---------
    1. Text A excerpt (with truncation note if needed)
    2. Text B excerpt (with truncation note if needed)
    3. The formatted evidence packet
    4. Optional: retrieved similar samples context

    Parameters
    ----------
    pair             : The TextPair containing text_a and text_b.
    evidence_packet  : Output of ``format_evidence_packet(features_a, features_b)``.
    retrieved_context: Optional pre-formatted string of retrieved stylistically
                       similar samples (for RAG step — not implemented yet in Step 4).
    """
    text_a, a_truncated = _truncate(pair.text_a)
    text_b, b_truncated = _truncate(pair.text_b)

    a_note = f"\n[...text truncated to {_TEXT_PREVIEW_CHARS} chars]" if a_truncated else ""
    b_note = f"\n[...text truncated to {_TEXT_PREVIEW_CHARS} chars]" if b_truncated else ""

    parts = [
        "=== TEXT A (Reference Document) ===",
        f"{text_a}{a_note}",
        "",
        "=== TEXT B (Unknown Document to Verify) ===",
        f"{text_b}{b_note}",
        "",
        evidence_packet,
    ]

    if retrieved_context:
        parts += [
            "",
            "=== RETRIEVED STYLISTICALLY SIMILAR SAMPLES ===",
            retrieved_context,
        ]

    return "\n".join(parts)


class AnalystAgent(BaseAgent):
    """Agent A — Stylometric Analyst.

    Makes the initial authorship verdict grounded strictly in the evidence
    packet.  In the full debate framework this agent runs first; its response
    is then passed to the Skeptic (Agent B).

    Parameters
    ----------
    model       : LLM model name.  Defaults to ``config.ANALYST_MODEL``
                  (set via ``ANALYST_MODEL`` in ``.env``).
    temperature : Sampling temperature.  Defaults to ``config.AGENT_TEMPERATURE``.
    prompt_path : Override the prompt file location (useful in tests).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        prompt_path: Optional[Path] = None,
    ) -> None:
        resolved_model = model or config.ANALYST_MODEL
        resolved_temp  = temperature if temperature is not None else config.AGENT_TEMPERATURE
        system_prompt  = _load_prompt(prompt_path or _PROMPT_PATH)

        super().__init__(
            model=resolved_model,
            system_prompt=system_prompt,
            temperature=resolved_temp,
        )

    def analyze(
        self,
        pair: TextPair,
        evidence_packet: str,
        retrieved_context: Optional[str] = None,
    ) -> AgentResponse:
        """Run the analyst on a single text pair.

        Parameters
        ----------
        pair              : The text pair to analyse.
        evidence_packet   : Pre-formatted string from ``format_evidence_packet``.
        retrieved_context : Optional RAG context string (pass ``None`` for Step 4;
                            will be populated when retrieval is implemented).

        Returns
        -------
        ``AgentResponse`` with the analyst's verdict, confidence, reasoning, and
        the list of feature names it cited.
        """
        user_message = _build_user_message(pair, evidence_packet, retrieved_context)
        return self.call(user_message)
