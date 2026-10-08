"""Debate orchestrator: controls the multi-agent debate flow and returns a DebateResult.

Flow
----
Round 1 (always):
    1. Analyst   → initial verdict (AgentResponse)
    2. Skeptic   → challenge       (SkepticResponse)
    3. Judge     → final verdict   (JudgeResponse)

Round 2 (when max_rounds >= 2):
    After the Skeptic challenges, the Analyst gets one rebuttal turn.
    The Judge then receives the full three-turn record before deciding.

The orchestrator is intentionally thin: it wires the agents together and
records the transcript.  All parsing, prompting, and LLM calls live in
the individual agent modules.

Public API
----------
    orchestrator = DebateOrchestrator(analyst, skeptic, judge, max_rounds=1)
    result = orchestrator.run(pair, evidence_packet)
    print(result.final_verdict, result.final_confidence)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from agents.analyst_agent import AnalystAgent
from agents.base_agent import AgentResponse
from agents.judge_agent import JudgeAgent, JudgeResponse
from agents.skeptic_agent import SkepticAgent, SkepticResponse
from data.preprocessor import TextPair


# ──────────────────────────────── Data Contract ───────────────────────────────

@dataclass
class DebateResult:
    """Complete record of a multi-agent debate.

    Fields
    ------
    analyst          : Agent A's initial verdict.
    skeptic          : Agent B's challenge.
    judge            : Agent C's final determination.
    rebuttal         : Agent A's round-2 rebuttal (None if max_rounds == 1).
    final_verdict    : Convenience alias for ``judge.verdict``.
    final_confidence : Convenience alias for ``judge.confidence``.
    elapsed_seconds  : Wall-clock time for the full debate.
    """

    analyst: AgentResponse
    skeptic: SkepticResponse
    judge: JudgeResponse
    rebuttal: Optional[AgentResponse]
    final_verdict: str
    final_confidence: float
    elapsed_seconds: float = field(default=0.0)


# ──────────────────────────────── Rebuttal Builder ────────────────────────────

_REBUTTAL_SYSTEM_NOTE = (
    "You previously made an authorship verdict. "
    "The Skeptic has now challenged your reasoning. "
    "Respond to their most important challenges with specific evidence from "
    "the packet. Update your verdict or confidence if warranted. "
    "Return the same format: VERDICT / CONFIDENCE / KEY FEATURES / REASONING."
)


def _build_rebuttal_message(
    pair: TextPair,
    evidence_packet: str,
    analyst_response: AgentResponse,
    skeptic_response: SkepticResponse,
) -> str:
    """Build the round-2 rebuttal prompt for the Analyst."""

    def _bullet(items: List[str]) -> str:
        return "\n".join(f"  - {i}" for i in items) if items else "  (none)"

    return "\n".join([
        "=== YOUR ORIGINAL VERDICT ===",
        f"Verdict    : {analyst_response.verdict}",
        f"Confidence : {analyst_response.confidence:.2f}",
        "",
        "Reasoning (excerpt):",
        analyst_response.reasoning[:800] + ("..." if len(analyst_response.reasoning) > 800 else ""),
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
        _REBUTTAL_SYSTEM_NOTE,
        "",
        "The evidence packet is reproduced below for reference.",
        "",
        evidence_packet,
    ])


# ──────────────────────────────── Orchestrator ────────────────────────────────

class DebateOrchestrator:
    """Controls the multi-agent debate flow.

    Parameters
    ----------
    analyst    : A configured ``AnalystAgent`` instance.
    skeptic    : A configured ``SkepticAgent`` instance.
    judge      : A configured ``JudgeAgent`` instance.
    max_rounds : 1 (default) runs Analyst → Skeptic → Judge.
                 2 adds an Analyst rebuttal between Skeptic and Judge.
    verbose    : If True, print progress lines to stdout during the debate.
    """

    def __init__(
        self,
        analyst: AnalystAgent,
        skeptic: SkepticAgent,
        judge: JudgeAgent,
        max_rounds: int = 1,
        verbose: bool = True,
    ) -> None:
        if max_rounds not in (1, 2):
            raise ValueError("max_rounds must be 1 or 2.")
        self.analyst = analyst
        self.skeptic = skeptic
        self.judge = judge
        self.max_rounds = max_rounds
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [debate] {msg}")

    def run(
        self,
        pair: TextPair,
        evidence_packet: str,
        retrieved_context: Optional[str] = None,
    ) -> DebateResult:
        """Run the full debate and return a ``DebateResult``.

        Parameters
        ----------
        pair              : The TextPair to analyse.
        evidence_packet   : Pre-formatted string from ``format_evidence_packet``.
        retrieved_context : Optional RAG context string passed to the Analyst.
                            Plugged in automatically once the retrieval step is
                            integrated.

        Returns
        -------
        ``DebateResult`` containing all agent responses and the final verdict.
        """
        t0 = time.perf_counter()

        # ── Round 1: Analyst makes initial verdict ────────────────────────────
        self._log("Round 1 — Analyst...")
        analyst_response = self.analyst.analyze(
            pair, evidence_packet, retrieved_context=retrieved_context
        )
        self._log(
            f"  Analyst → {analyst_response.verdict} "
            f"(conf={analyst_response.confidence:.2f}, parse_ok={analyst_response.parse_ok})"
        )

        # ── Round 1: Skeptic challenges ───────────────────────────────────────
        self._log("Round 1 — Skeptic...")
        skeptic_response = self.skeptic.challenge(
            pair, evidence_packet, analyst_response
        )
        self._log(
            f"  Skeptic → {skeptic_response.stance} "
            f"(conf={skeptic_response.confidence:.2f}, parse_ok={skeptic_response.parse_ok})"
        )

        # ── Round 2 (optional): Analyst rebuts ───────────────────────────────
        rebuttal: Optional[AgentResponse] = None
        if self.max_rounds >= 2:
            self._log("Round 2 — Analyst rebuttal...")
            rebuttal_msg = _build_rebuttal_message(
                pair, evidence_packet, analyst_response, skeptic_response
            )
            rebuttal = self.analyst.call(rebuttal_msg)
            self._log(
                f"  Rebuttal → {rebuttal.verdict} "
                f"(conf={rebuttal.confidence:.2f}, parse_ok={rebuttal.parse_ok})"
            )

        # ── Judge integrates and decides ──────────────────────────────────────
        self._log("Judge — final verdict...")
        judge_response = self.judge.adjudicate(
            pair, evidence_packet, analyst_response, skeptic_response, rebuttal
        )
        self._log(
            f"  Judge → {judge_response.verdict} "
            f"(conf={judge_response.confidence:.2f}, agreement={judge_response.agent_agreement}, "
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
