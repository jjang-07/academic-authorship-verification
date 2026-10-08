"""Text-only debate orchestrator: all three agents reason from raw text alone.

Flow
----
Round 1 (always):
    1. AnalystAgentTextOnly   → initial verdict (AgentResponse)
    2. SkepticAgentTextOnly   → close-reading challenge (SkepticResponse)
    3. JudgeAgentTextOnly     → final verdict (JudgeResponse)

Round 2 (when max_rounds >= 2):
    After the Skeptic's challenge, the Analyst receives a rebuttal prompt
    containing both texts plus the Skeptic's challenge (no evidence packet).
    The Judge then receives the full four-turn record.

Key difference from ``DebateOrchestrator``
------------------------------------------
No evidence packet is passed anywhere in this pipeline.  Every agent receives
only raw text excerpts and the debate transcript so far.  This keeps the
comparison with the feature-based pipeline clean: the only information source
is the prose itself.

The return type is the same ``DebateResult`` dataclass, so all downstream
evaluation code (``evaluation/ablation.py``, ``evaluation/metrics.py``, etc.)
works without modification.

Public API
----------
    orchestrator = DebateOrchestratorTextOnly(
        analyst, skeptic, judge, max_rounds=1, verbose=True,
        dataset_context="student_essays",  # optional
    )
    result = orchestrator.run(pair)   # no evidence_packet argument
    print(result.final_verdict, result.final_confidence)
"""

from __future__ import annotations

import time
from typing import Optional

from agents.analyst_agent_textonly import AnalystAgentTextOnly
from agents.base_agent import AgentResponse
from agents.judge_agent_textonly import JudgeAgentTextOnly
from agents.skeptic_agent_textonly import SkepticAgentTextOnly
from data.preprocessor import TextPair
from debate.orchestrator import DebateResult
from debate.textonly_dataset_prompt import apply_textonly_dataset_context

# Characters shown per text in the rebuttal prompt (matches agent defaults).
_TEXT_PREVIEW_CHARS = 2500

# System note injected before the rebuttal texts.
_REBUTTAL_NOTE = (
    "You previously made an authorship verdict based on close reading. "
    "The Skeptic has now challenged your reasoning. "
    "Re-read both texts below and respond to the Skeptic's most important "
    "challenges with specific observations from the texts. "
    "Update your verdict or confidence if warranted. "
    "Return the same format: VERDICT / CONFIDENCE / KEY FEATURES / REASONING."
)


def _build_rebuttal_message(
    pair: TextPair,
    analyst_response: AgentResponse,
    skeptic_response,          # SkepticResponse
) -> str:
    """Build the round-2 rebuttal prompt for the Analyst (text-only version).

    Includes the original verdict, the Skeptic's challenge, and both raw texts
    for re-reading.  No evidence packet.
    """
    def _trunc(text: str) -> tuple[str, bool]:
        if len(text) <= _TEXT_PREVIEW_CHARS:
            return text, False
        return text[:_TEXT_PREVIEW_CHARS], True

    text_a, a_trunc = _trunc(pair.text_a)
    text_b, b_trunc = _trunc(pair.text_b)
    a_note = f"\n[...truncated to {_TEXT_PREVIEW_CHARS} chars]" if a_trunc else ""
    b_note = f"\n[...truncated to {_TEXT_PREVIEW_CHARS} chars]" if b_trunc else ""

    def _bullet(items) -> str:
        return "\n".join(f"  - {i}" for i in items) if items else "  (none)"

    return "\n".join([
        "=== YOUR ORIGINAL VERDICT ===",
        f"Verdict    : {analyst_response.verdict}",
        f"Confidence : {analyst_response.confidence:.2f}",
        "",
        "Reasoning (excerpt):",
        analyst_response.reasoning[:800] + (
            "..." if len(analyst_response.reasoning) > 800 else ""
        ),
        "",
        "=== SKEPTIC'S CHALLENGE ===",
        f"Stance     : {skeptic_response.stance}",
        f"Confidence : {skeptic_response.confidence:.2f}",
        "",
        "Challenges:",
        _bullet(skeptic_response.challenges),
        "",
        "Overlooked evidence the Skeptic identified:",
        _bullet(skeptic_response.overlooked_evidence),
        "",
        "=== YOUR TASK ===",
        _REBUTTAL_NOTE,
        "",
        "=== TEXT A (re-read before responding) ===",
        f"{text_a}{a_note}",
        "",
        "=== TEXT B (re-read before responding) ===",
        f"{text_b}{b_note}",
    ])


class DebateOrchestratorTextOnly:
    """Controls the fully text-only multi-agent debate flow.

    Parameters
    ----------
    analyst    : A configured ``AnalystAgentTextOnly`` instance.
    skeptic    : A configured ``SkepticAgentTextOnly`` instance.
    judge      : A configured ``JudgeAgentTextOnly`` instance.
    max_rounds : 1 (default) runs Analyst → Skeptic → Judge.
                 2 adds an Analyst rebuttal between Skeptic and Judge.
    verbose    : If True, print progress lines to stdout during the debate.
    dataset_context : Optional corpus tag (e.g. ``"student_essays"``) to inject
                 dataset-specific system-prompt sections on all three agents.
                 See ``debate.textonly_dataset_prompt``.
    """

    def __init__(
        self,
        analyst: AnalystAgentTextOnly,
        skeptic: SkepticAgentTextOnly,
        judge: JudgeAgentTextOnly,
        max_rounds: int = 1,
        verbose: bool = True,
        dataset_context: Optional[str] = None,
    ) -> None:
        if max_rounds not in (1, 2):
            raise ValueError("max_rounds must be 1 or 2.")
        self.analyst = analyst
        self.skeptic = skeptic
        self.judge = judge
        self.max_rounds = max_rounds
        self.verbose = verbose
        self.dataset_context = dataset_context

        if dataset_context is not None:
            self.analyst.system_prompt = apply_textonly_dataset_context(
                self.analyst.system_prompt, "analyst", dataset_context
            )
            self.skeptic.system_prompt = apply_textonly_dataset_context(
                self.skeptic.system_prompt, "skeptic", dataset_context
            )
            self.judge.system_prompt = apply_textonly_dataset_context(
                self.judge.system_prompt, "judge", dataset_context
            )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [debate-textonly] {msg}")

    def run(self, pair: TextPair) -> DebateResult:
        """Run the full text-only debate and return a ``DebateResult``.

        Parameters
        ----------
        pair : The ``TextPair`` to analyse.  No evidence packet is required.

        Returns
        -------
        ``DebateResult`` — same dataclass as ``DebateOrchestrator.run()``,
        fully compatible with all evaluation and interpretability code.
        """
        t0 = time.perf_counter()

        # ── Round 1: Analyst reads both texts and makes initial verdict ────────
        self._log("Round 1 — Analyst (text-only)...")
        analyst_response = self.analyst.analyze(pair)
        self._log(
            f"  Analyst → {analyst_response.verdict} "
            f"(conf={analyst_response.confidence:.2f}, parse_ok={analyst_response.parse_ok})"
        )

        # ── Round 1: Skeptic re-reads both texts and challenges ───────────────
        self._log("Round 1 — Skeptic (text-only)...")
        skeptic_response = self.skeptic.challenge(pair, analyst_response)
        self._log(
            f"  Skeptic → {skeptic_response.stance} "
            f"(conf={skeptic_response.confidence:.2f}, parse_ok={skeptic_response.parse_ok})"
        )

        # ── Round 2 (optional): Analyst re-reads and rebuts ──────────────────
        rebuttal: Optional[AgentResponse] = None
        if self.max_rounds >= 2:
            self._log("Round 2 — Analyst rebuttal (text-only)...")
            rebuttal_msg = _build_rebuttal_message(
                pair, analyst_response, skeptic_response
            )
            rebuttal = self.analyst.call(rebuttal_msg)
            self._log(
                f"  Rebuttal → {rebuttal.verdict} "
                f"(conf={rebuttal.confidence:.2f}, parse_ok={rebuttal.parse_ok})"
            )

        # ── Judge re-reads both texts and the full debate transcript ──────────
        self._log("Judge — final verdict (text-only)...")
        judge_response = self.judge.adjudicate(
            pair, analyst_response, skeptic_response, rebuttal
        )
        self._log(
            f"  Judge → {judge_response.verdict} "
            f"(conf={judge_response.confidence:.2f}, "
            f"agreement={judge_response.agent_agreement}, "
            f"parse_ok={judge_response.parse_ok})"
        )

        elapsed = time.perf_counter() - t0

        return DebateResult(
            analyst=analyst_response,
            skeptic=skeptic_response,
            judge=judge_response,
            rebuttal=rebuttal,
            final_verdict=judge_response.verdict,
            final_confidence=judge_response.confidence,
            elapsed_seconds=elapsed,
        )
