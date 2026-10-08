#!/usr/bin/env python3
"""AI INK — Authorship Verification Demo
Flask backend. Run from repo root:
    ./venv/bin/python web/app.py
    ./venv/bin/python web/app.py --port 5000

Default port is **5050** because macOS often reserves **5000** for AirPlay Receiver.

Override: ``AI_INK_PORT=5001`` or ``PORT=5001`` in the environment, or ``--port``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config

# ── Startup .env check ────────────────────────────────────────────────────────
if not config.OPENAI_API_KEY:
    print(
        "\n[AI INK] ERROR: OPENAI_API_KEY is not set.\n"
        "  Add it to .env in the repo root before launching.\n",
        file=sys.stderr,
    )
    sys.exit(1)

from flask import Flask, jsonify, render_template, request  # noqa: E402

app = Flask(__name__, template_folder="templates")
app.config["JSON_SORT_KEYS"] = False


# ── Lazy agent / orchestrator imports (heavy; deferred until first request) ──

def _make_pair(text_a: str, text_b: str):
    from data.preprocessor import TextPair
    return TextPair(
        text_a=text_a,
        text_b=text_b,
        label=0,
        author_id="demo",
        topic_a="unknown",
        topic_b="unknown",
        source_dataset="demo",
    )


def run_single(text_a: str, text_b: str) -> dict:
    from agents.analyst_agent_textonly import AnalystAgentTextOnly
    pair = _make_pair(text_a, text_b)
    analyst = AnalystAgentTextOnly()
    t0 = time.perf_counter()
    resp = analyst.analyze(pair)
    elapsed = round(time.perf_counter() - t0, 2)
    return {
        "mode": "single",
        "verdict": resp.verdict,
        "confidence": round(resp.confidence, 3),
        "analyst": {
            "verdict": resp.verdict,
            "confidence": round(resp.confidence, 3),
            "reasoning": resp.reasoning or "",
            "key_features": resp.key_features_cited or [],
            "parse_ok": resp.parse_ok,
        },
        "elapsed_seconds": elapsed,
    }


def run_debate(text_a: str, text_b: str) -> dict:
    from agents.analyst_agent_textonly import AnalystAgentTextOnly
    from agents.judge_agent_textonly import JudgeAgentTextOnly
    from agents.skeptic_agent_textonly import SkepticAgentTextOnly
    from debate.orchestrator_textonly import DebateOrchestratorTextOnly

    pair = _make_pair(text_a, text_b)
    analyst = AnalystAgentTextOnly()
    skeptic = SkepticAgentTextOnly()
    judge = JudgeAgentTextOnly()
    orch = DebateOrchestratorTextOnly(analyst, skeptic, judge, verbose=False)

    t0 = time.perf_counter()
    result = orch.run(pair)
    elapsed = round(time.perf_counter() - t0, 2)

    a = result.analyst
    s = result.skeptic
    j = result.judge

    return {
        "mode": "debate",
        "verdict": result.final_verdict,
        "confidence": round(result.final_confidence, 3),
        "analyst": {
            "verdict": a.verdict,
            "confidence": round(a.confidence, 3),
            "reasoning": a.reasoning or "",
            "key_features": a.key_features_cited or [],
            "parse_ok": a.parse_ok,
        },
        "skeptic": {
            "stance": s.stance,
            "confidence": round(s.confidence, 3),
            "challenges": s.challenges or [],
            "overlooked_evidence": s.overlooked_evidence or [],
            "revised_reasoning": s.revised_reasoning or "",
            "parse_ok": s.parse_ok,
        },
        "judge": {
            "verdict": j.verdict,
            "confidence": round(j.confidence, 3),
            "decisive_factors": j.decisive_factors or [],
            "educator_summary": j.educator_summary or "",
            "agent_agreement": j.agent_agreement,
            "parse_ok": j.parse_ok,
        },
        "elapsed_seconds": elapsed,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    text_a = (data.get("text_a") or "").strip()
    text_b = (data.get("text_b") or "").strip()
    mode = (data.get("mode") or "single").strip().lower()

    if not text_a or not text_b:
        return jsonify({"error": "Both text_a and text_b are required."}), 400
    if len(text_a) < 20 or len(text_b) < 20:
        return jsonify({"error": "Texts must each be at least 20 characters."}), 400

    try:
        if mode == "debate":
            result = run_debate(text_a, text_b)
        else:
            result = run_single(text_a, text_b)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def _resolve_port(cli_port: int | None) -> int:
    """Port: CLI --port > AI_INK_PORT > PORT > default 5050 (avoids macOS AirPlay on 5000)."""
    if cli_port is not None:
        return cli_port
    for key in ("AI_INK_PORT", "PORT"):
        raw = os.environ.get(key, "").strip()
        if raw.isdigit():
            return int(raw)
    return 5050


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AI INK demo server")
    ap.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="N",
        help="Listen port (default: 5050, or AI_INK_PORT / PORT env).",
    )
    args = ap.parse_args()
    port = _resolve_port(args.port)

    print("\n" + "=" * 54)
    print("  AI INK — Authorship Verification Demo")
    print(f"  http://localhost:{port}")
    print("=" * 54 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port)
