"""Text-only Analyst agent — close reading without any feature statistics.

Differences from AnalystAgent and AnalystAgentICL
--------------------------------------------------
- The user message contains **only the two raw text excerpts** — no evidence
  packet, no feature numbers, no delta labels, no severity thresholds.
- Reasoning is grounded purely in close reading along six stylistic dimensions
  (sentence rhythm, punctuation habits, formality & register, transition
  phrases, syntactic preferences, vocabulary register) defined in the prompt.
- Four in-context learning examples in the system prompt show how to reason
  from prose alone, including cases where surface register differs but the
  underlying voice is the same (the hardest cross-topic case).

Compatible surface
------------------
- Output format is **identical** to AnalystAgent (VERDICT / CONFIDENCE /
  KEY FEATURES / REASONING headers), so AgentResponse parsing and all
  downstream Skeptic / Judge / DebateOrchestrator code are unchanged.
- Constructor and ``analyze()`` signature mirror AnalystAgent, making the
  three analyst implementations fully interchangeable in tests.

Public API
----------
    agent    = AnalystAgentTextOnly()
    response = agent.analyze(pair)          # no features, no packet needed
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import config
from agents.base_agent import AgentResponse, BaseAgent
from data.preprocessor import TextPair

_TEXTONLY_PROMPT_PATH = Path(__file__).parent / "prompts" / "analyst_prompt_textonly.txt"

# Same ceiling as the other analyst variants.
_TEXT_PREVIEW_CHARS = 2500


def _load_textonly_prompt(path: Path = _TEXTONLY_PROMPT_PATH) -> str:
    """Read the text-only system prompt from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Text-only analyst prompt not found: {path}\n"
            "Expected location: agents/prompts/analyst_prompt_textonly.txt"
        )
    return path.read_text(encoding="utf-8").strip()


def _truncate(text: str, max_chars: int = _TEXT_PREVIEW_CHARS) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _build_user_message(pair: TextPair) -> str:
    """Assemble the user message: two raw text excerpts, nothing else.

    Format matches the input format described in the system prompt:

        TEXT A:
        [full text]

        TEXT B:
        [full text]
    """
    text_a, a_truncated = _truncate(pair.text_a)
    text_b, b_truncated = _truncate(pair.text_b)

    a_note = f"\n[...truncated to {_TEXT_PREVIEW_CHARS} chars]" if a_truncated else ""
    b_note = f"\n[...truncated to {_TEXT_PREVIEW_CHARS} chars]" if b_truncated else ""

    return "\n".join([
        "TEXT A:",
        f"{text_a}{a_note}",
        "",
        "TEXT B:",
        f"{text_b}{b_note}",
    ])


class AnalystAgentTextOnly(BaseAgent):
    """Text-only variant of the Analyst (Agent A).

    Performs authorship analysis by close reading alone — no numeric features,
    no delta annotations.  Uses six stylistic dimensions and four ICL examples
    to guide reasoning from raw prose.

    Parameters
    ----------
    model       : LLM model name.  Defaults to ``config.ANALYST_MODEL``.
    temperature : Sampling temperature.  Defaults to ``config.AGENT_TEMPERATURE``.
    prompt_path : Override the prompt file location (useful in tests).
    system_prompt : If set, use this system text instead of loading ``prompt_path``.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        prompt_path: Optional[Path] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        resolved_model = model or config.ANALYST_MODEL
        resolved_temp  = temperature if temperature is not None else config.AGENT_TEMPERATURE
        if system_prompt is not None:
            sp = system_prompt.strip()
        else:
            sp = _load_textonly_prompt(prompt_path or _TEXTONLY_PROMPT_PATH)

        super().__init__(
            model=resolved_model,
            system_prompt=sp,
            temperature=resolved_temp,
        )

    def analyze(self, pair: TextPair) -> AgentResponse:
        """Run the text-only analyst on a single text pair.

        Parameters
        ----------
        pair : The text pair to analyse.  Only ``pair.text_a`` and
               ``pair.text_b`` are used — no feature dicts required.

        Returns
        -------
        ``AgentResponse`` with the same fields as AnalystAgent —
        verdict, confidence, key_features_cited, reasoning, raw_response, parse_ok.
        """
        return self.call(_build_user_message(pair))
