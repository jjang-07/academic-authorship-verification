"""In-Context Learning (ICL) Analyst agent — an alternative to AnalystAgent.

Differences from the standard AnalystAgent
-------------------------------------------
- The user message presents **raw text excerpts** for both samples rather than
  a structured numeric evidence packet.
- The feature summary contains only the five most diagnostic topic-independent
  features (passive_voice_freq, coord_subord_ratio, contraction_per_100,
  hedging_per_100, sent_opening_pronoun) with raw values only — no delta
  severity labels (no LOW / MODERATE / HIGH tags).
- The system prompt contains four balanced in-context learning examples that
  teach the model how to reason from micro-habit features directly.

Compatible surface
------------------
- The output format is **identical** to AnalystAgent (VERDICT / CONFIDENCE /
  KEY FEATURES / REASONING headers), so the same AgentResponse parser and all
  downstream Skeptic / Judge / DebateOrchestrator code work without changes.
- The constructor and ``analyze()`` signature are the same as AnalystAgent,
  making the two agents fully interchangeable in ``DebateOrchestrator``.

Public API
----------
    agent    = AnalystAgentICL()              # uses ANALYST_MODEL from config
    response = agent.analyze(pair, features_a, features_b)   # AgentResponse
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import config
from agents.base_agent import AgentResponse, BaseAgent
from data.preprocessor import TextPair

# Path to the ICL system prompt relative to this file.
_ICL_PROMPT_PATH = Path(__file__).parent / "prompts" / "analyst_prompt_icl.txt"

# Characters to show per text in the user message — same ceiling as AnalystAgent.
_TEXT_PREVIEW_CHARS = 2500

# The five topic-independent micro-habit features delivered to the model.
_ICL_FEATURES: tuple[str, ...] = (
    "passive_voice_freq",
    "coord_subord_ratio",
    "contraction_per_100",
    "hedging_per_100",
    "sent_opening_pronoun",
)

# Human-readable labels for the feature summary.
_FEATURE_LABELS: dict[str, str] = {
    "passive_voice_freq":   "passive_voice_freq     (passive constructions per sentence)",
    "coord_subord_ratio":   "coord_subord_ratio     (coordinating / subordinating conjunctions)",
    "contraction_per_100":  "contraction_per_100    (contractions per 100 words)",
    "hedging_per_100":      "hedging_per_100        (hedging terms per 100 words)",
    "sent_opening_pronoun": "sent_opening_pronoun   (fraction of sentences opening with a pronoun)",
}


def _load_icl_prompt(path: Path = _ICL_PROMPT_PATH) -> str:
    """Read the ICL system prompt from disk.

    Raises
    ------
    FileNotFoundError
        If the ICL prompt file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"ICL analyst prompt not found: {path}\n"
            "Expected location: agents/prompts/analyst_prompt_icl.txt"
        )
    return path.read_text(encoding="utf-8").strip()


def _truncate(text: str, max_chars: int = _TEXT_PREVIEW_CHARS) -> tuple[str, bool]:
    """Return (possibly truncated text, was_truncated)."""
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _build_feature_summary(
    features_a: Dict[str, Any],
    features_b: Dict[str, Any],
) -> str:
    """Format the five micro-habit features into a plain-English summary.

    Output format (no delta severity labels — raw values only):

        MICRO-HABIT FEATURE SUMMARY (topic-independent):
          passive_voice_freq    — Text A: 0.18, Text B: 0.22
          coord_subord_ratio    — Text A: 1.73, Text B: 1.65
          ...
    """
    lines = ["=== MICRO-HABIT FEATURE SUMMARY (topic-independent) ==="]
    for key in _ICL_FEATURES:
        label = _FEATURE_LABELS.get(key, key)
        val_a = features_a.get(key)
        val_b = features_b.get(key)

        if val_a is None and val_b is None:
            lines.append(f"  {label} — [not available]")
            continue

        str_a = f"{val_a:.3f}" if isinstance(val_a, float) else str(val_a)
        str_b = f"{val_b:.3f}" if isinstance(val_b, float) else str(val_b)
        lines.append(f"  {label} — Text A: {str_a}, Text B: {str_b}")

    return "\n".join(lines)


def _build_user_message(
    pair: TextPair,
    features_a: Dict[str, Any],
    features_b: Dict[str, Any],
) -> str:
    """Assemble the full user message for the ICL analyst.

    Structure
    ---------
    1. Text A excerpt (with truncation note if needed)
    2. Text B excerpt (with truncation note if needed)
    3. Micro-habit feature summary (raw values, no delta labels)

    Note: The ICL agent does NOT receive a structured evidence packet — only
    the five micro-habit features are exposed to keep the comparison clean.
    """
    text_a, a_truncated = _truncate(pair.text_a)
    text_b, b_truncated = _truncate(pair.text_b)

    a_note = f"\n[...text truncated to {_TEXT_PREVIEW_CHARS} chars]" if a_truncated else ""
    b_note = f"\n[...text truncated to {_TEXT_PREVIEW_CHARS} chars]" if b_truncated else ""

    feature_summary = _build_feature_summary(features_a, features_b)

    parts = [
        "=== TEXT A (Reference Document) ===",
        f"{text_a}{a_note}",
        "",
        "=== TEXT B (Unknown Document to Verify) ===",
        f"{text_b}{b_note}",
        "",
        feature_summary,
        "",
        "Now apply the SAME reasoning pattern as the examples above to produce "
        "your verdict for this pair.",
    ]

    return "\n".join(parts)


class AnalystAgentICL(BaseAgent):
    """ICL variant of the Analyst (Agent A).

    Receives raw text excerpts and a five-feature natural-language summary
    instead of a structured delta-annotated evidence packet.  The in-context
    learning examples in the system prompt guide the model to reason from
    micro-habit feature values directly.

    The output format is identical to AnalystAgent so this class is a
    drop-in replacement inside DebateOrchestrator.

    Parameters
    ----------
    model       : LLM model name.  Defaults to ``config.ANALYST_MODEL``.
    temperature : Sampling temperature.  Defaults to ``config.AGENT_TEMPERATURE``.
    prompt_path : Override the ICL prompt file location (useful in tests).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        prompt_path: Optional[Path] = None,
    ) -> None:
        resolved_model = model or config.ANALYST_MODEL
        resolved_temp  = temperature if temperature is not None else config.AGENT_TEMPERATURE
        system_prompt  = _load_icl_prompt(prompt_path or _ICL_PROMPT_PATH)

        super().__init__(
            model=resolved_model,
            system_prompt=system_prompt,
            temperature=resolved_temp,
        )

    def analyze(
        self,
        pair: TextPair,
        features_a: Dict[str, Any],
        features_b: Dict[str, Any],
    ) -> AgentResponse:
        """Run the ICL analyst on a single text pair.

        Parameters
        ----------
        pair       : The text pair to analyse.
        features_a : Full feature dict for Text A (output of ``extract_all``).
        features_b : Full feature dict for Text B (output of ``extract_all``).

        Returns
        -------
        ``AgentResponse`` with the same fields as the standard AnalystAgent —
        verdict, confidence, key_features_cited, reasoning, raw_response, parse_ok.
        """
        user_message = _build_user_message(pair, features_a, features_b)
        return self.call(user_message)
