"""Base LLM agent: API routing, response parsing, and the AgentResponse dataclass.

Routing logic
-------------
Model name prefix determines which provider SDK is used:

    gpt-* | o1-* | o3-* | o4-*  →  OpenAI   (openai>=1.0)
    claude-*              →  Anthropic (anthropic>=0.20)

Everything else raises ``ValueError`` with a clear message so mis-spellings in
``.env`` are caught immediately rather than silently falling back.

API keys are read from ``config`` (never hardcoded).  If the required key is
absent from the environment, a ``ValueError`` is raised at call time — not at
import time — so modules that import this file without making API calls never
fail due to missing credentials.
"""

from __future__ import annotations

import logging
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import List

import config

_LOG = logging.getLogger(__name__)

# Retry backoff: 10s, 20s, 40s, 80s, 160s (5 retries)
_RETRY_DELAYS = [10, 20, 40, 80, 160]
_MAX_RETRIES = len(_RETRY_DELAYS)
_RETRYABLE_STATUS_CODES = (429, 500, 503)


def _get_status_code(exc: BaseException) -> int | None:
    """Extract HTTP status code from OpenAI/Anthropic API exceptions."""
    if hasattr(exc, "status_code"):
        return getattr(exc, "status_code", None)
    if hasattr(exc, "response") and exc.response is not None:
        return getattr(exc.response, "status_code", None)
    if hasattr(exc, "http_status"):
        return getattr(exc, "http_status", None)
    return None


def _is_retryable(exc: BaseException) -> bool:
    """Return True if the exception indicates a retryable error (429, 500, 503)."""
    code = _get_status_code(exc)
    return code in _RETRYABLE_STATUS_CODES if code is not None else False


# ──────────────────────────────── Data Contract ───────────────────────────────

@dataclass
class AgentResponse:
    """Structured output returned by every agent ``call()`` invocation.

    Fields
    ------
    verdict            : "SAME_AUTHOR", "DIFFERENT_AUTHOR", or "UNKNOWN" if
                         parsing failed.
    confidence         : Float in [0.0, 1.0].  Defaults to 0.5 on parse failure.
    reasoning          : Multi-paragraph natural language explanation from the LLM.
    key_features_cited : Which feature names the agent cited as most decisive.
    raw_response       : Complete unmodified LLM output, preserved for logging and
                         debugging.
    parse_ok           : True when all four structured fields were parsed cleanly.
                         False indicates the LLM deviated from the required format.
    """

    verdict: str
    confidence: float
    reasoning: str
    key_features_cited: List[str]
    raw_response: str
    parse_ok: bool = field(default=True)


# ──────────────────────────────── Response Parser ─────────────────────────────

_VERDICT_RE   = re.compile(
    r"VERDICT\s*:\s*\[?\s*(SAME_AUTHOR|DIFFERENT_AUTHOR)\s*\]?",
    re.IGNORECASE,
)
_CONFIDENCE_RE = re.compile(
    r"CONFIDENCE\s*:\s*\[?\s*([0-9]*\.?[0-9]+)\s*\]?",
)
_FEATURES_RE = re.compile(
    r"KEY\s+FEATURES\s*:\s*\[?(.*?)\]?\s*(?=\nREASONING\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)
_REASONING_RE = re.compile(
    r"REASONING\s*:\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)


def _parse_agent_response(raw: str) -> AgentResponse:
    """Extract structured fields from a raw LLM text output.

    Designed to be tolerant of minor formatting deviations while failing loudly
    enough (via ``parse_ok=False``) that callers can detect format drift.
    """
    parse_ok = True

    # ── VERDICT ──────────────────────────────────────────────────────────────
    vm = _VERDICT_RE.search(raw)
    if vm:
        verdict = vm.group(1).upper()
    else:
        verdict = "UNKNOWN"
        parse_ok = False
        warnings.warn(
            "AgentResponse parse: could not find VERDICT field. "
            "LLM may have deviated from the required output format.",
            stacklevel=3,
        )

    # ── CONFIDENCE ───────────────────────────────────────────────────────────
    cm = _CONFIDENCE_RE.search(raw)
    if cm:
        raw_conf = float(cm.group(1))
        # The LLM occasionally outputs values like "73" instead of "0.73"
        confidence = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf
        confidence = max(0.0, min(1.0, confidence))
    else:
        confidence = 0.5
        parse_ok = False

    # ── KEY FEATURES ─────────────────────────────────────────────────────────
    fm = _FEATURES_RE.search(raw)
    if fm:
        raw_features = fm.group(1).strip()
        # Accept comma-separated, semicolon-separated, or dash-prefixed lists
        parts = re.split(r"[,;\n]", raw_features)
        key_features_cited = [
            p.strip().lstrip("-• ").strip()
            for p in parts
            if p.strip().lstrip("-• ").strip()
        ]
    else:
        key_features_cited = []
        parse_ok = False

    # ── REASONING ────────────────────────────────────────────────────────────
    rm = _REASONING_RE.search(raw)
    if rm:
        reasoning = rm.group(1).strip()
    else:
        # Fall back: everything after the last matched section header
        reasoning = raw.strip()
        parse_ok = False

    return AgentResponse(
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
        key_features_cited=key_features_cited,
        raw_response=raw,
        parse_ok=parse_ok,
    )


# ──────────────────────────────── BaseAgent ───────────────────────────────────

def _is_openai_responses_model(model: str) -> bool:
    """Return True for models that require the Responses API (/v1/responses).

    Any model whose name starts with ``gpt-5`` is routed here.  This covers:
      • gpt-5.4        (full reasoning model, reasoning_effort=high)
      • gpt-5.4-mini   (smaller reasoning model, reasoning_effort=high)

    These models do not support the Chat Completions endpoint and must be
    called via ``client.responses.create()``.  Only ``gpt-5*`` models receive
    ``reasoning={"effort": "high"}``; other future Responses-routed models can
    omit that parameter (e.g. ``o4-mini`` uses Chat Completions instead).
    """
    return model.lower().startswith("gpt-5")


def _openai_responses_include_reasoning_effort(model: str) -> bool:
    """Only gpt-5* models accept ``reasoning`` on the Responses API in our stack."""
    return model.lower().startswith("gpt-5")


def _openai_chat_accepts_temperature(model: str) -> bool:
    """o-series reasoning models often reject custom ``temperature`` on Chat Completions."""
    m = model.lower()
    return not (m.startswith("o1-") or m.startswith("o3-") or m.startswith("o4-"))


def _is_openai_model(model: str) -> bool:
    """Return True when the model name belongs to an OpenAI model family."""
    prefixes = ("gpt-", "o1-", "o3-", "o4-", "text-", "davinci", "babbage",
                "curie", "ada", "chatgpt")
    return any(model.lower().startswith(p) for p in prefixes)


def _is_anthropic_model(model: str) -> bool:
    """Return True when the model name belongs to an Anthropic model family."""
    return model.lower().startswith("claude")


class BaseAgent:
    """Provider-agnostic LLM agent.

    Usage
    -----
        agent = BaseAgent(model="gpt-4o-mini", system_prompt="You are ...")
        response: AgentResponse = agent.call("User message here")

    The constructor never makes network calls.  API imports are deferred to
    ``call()`` so that the module can be imported even when the optional LLM
    SDKs are not installed (useful for unit tests that mock ``call``).
    """

    def __init__(
        self,
        model: str,
        system_prompt: str,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature

        if not _is_openai_model(model) and not _is_anthropic_model(model):
            raise ValueError(
                f"Unrecognised model '{model}'. "
                "Expected a model starting with 'gpt-', 'o1-', 'o3-' (OpenAI) "
                "or 'claude-' (Anthropic).  "
                "Update ANALYST_MODEL / SKEPTIC_MODEL / JUDGE_MODEL in your .env."
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _call_openai_responses(self, user_message: str) -> str:
        """Call the OpenAI Responses API (/v1/responses) with reasoning=high.

        Used for gpt-5* models, which do not support the Chat Completions
        endpoint.  The system prompt is passed as a ``system``-role item in
        the ``input`` array.  ``reasoning={"effort": "high"}`` is always set;
        ``temperature`` is intentionally omitted because reasoning models
        control their own sampling.
        """
        if not config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file before calling an OpenAI model."
            )
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "openai package not found. Install it with: pip install openai"
            ) from exc

        client = OpenAI(api_key=config.OPENAI_API_KEY)
        create_kwargs: dict = {
            "model": self.model,
            "input": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        if _openai_responses_include_reasoning_effort(self.model):
            create_kwargs["reasoning"] = {"effort": "high"}
        response = client.responses.create(**create_kwargs)
        return response.output_text or ""

    def _call_openai(self, user_message: str) -> str:
        """Call the OpenAI Chat Completions API and return raw text."""
        if not config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file before calling an OpenAI model."
            )
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "openai package not found. Install it with: pip install openai"
            ) from exc

        client = OpenAI(api_key=config.OPENAI_API_KEY)
        chat_kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        if _openai_chat_accepts_temperature(self.model):
            chat_kwargs["temperature"] = self.temperature
        completion = client.chat.completions.create(**chat_kwargs)
        return completion.choices[0].message.content or ""

    def _call_anthropic(self, user_message: str) -> str:
        """Call the Anthropic Messages API and return raw text."""
        if not config.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file before calling an Anthropic model."
            )
        try:
            from anthropic import Anthropic  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "anthropic package not found. Install it with: pip install anthropic"
            ) from exc

        client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        # Anthropic returns a list of content blocks; join text blocks
        return "".join(
            block.text for block in message.content if hasattr(block, "text")
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def _raw_call(self, user_message: str) -> str:
        """Call the LLM and return the raw text without any parsing.

        Retries up to 5 times on HTTP 429 (rate limit), 500, 503 with exponential
        backoff (10s, 20s, 40s, 80s, 160s). Logs each retry attempt.

        Routing:
          gpt-5*  → Responses API  (_call_openai_responses; reasoning=high only for gpt-5*)
          gpt-*/o* → Chat Completions (_call_openai; no custom temperature for o1/o3/o4)
          claude-* → Anthropic Messages (_call_anthropic)

        Subclasses (SkepticAgent, JudgeAgent) use this to apply their own
        response parsers instead of the analyst-specific ``_parse_agent_response``.
        """
        def _do_call() -> str:
            if _is_openai_responses_model(self.model):
                return self._call_openai_responses(user_message)
            if _is_openai_model(self.model):
                return self._call_openai(user_message)
            return self._call_anthropic(user_message)

        for attempt in range(_MAX_RETRIES + 1):
            try:
                return _do_call()
            except BaseException as exc:
                if attempt < _MAX_RETRIES and _is_retryable(exc):
                    delay = _RETRY_DELAYS[attempt]
                    code = _get_status_code(exc)
                    _LOG.warning(
                        "API call failed (HTTP %s), retry %d/%d in %ds: %s",
                        code or "?",
                        attempt + 1,
                        _MAX_RETRIES,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                else:
                    raise

    def call(self, user_message: str) -> AgentResponse:
        """Send ``user_message`` to the configured LLM and return a parsed response.

        Raises
        ------
        ValueError
            Missing API key, or unrecognised model name.
        ImportError
            Required SDK (openai / anthropic) not installed.
        Any provider-specific exception
            Network errors, rate limits, etc. are propagated as-is so callers
            can decide how to handle retries.
        """
        return _parse_agent_response(self._raw_call(user_message))
