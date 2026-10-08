"""Agents package: LLM agent classes for the V2 debate framework."""

from agents.base_agent import AgentResponse, BaseAgent
from agents.analyst_agent import AnalystAgent
from agents.skeptic_agent import SkepticAgent, SkepticResponse
from agents.judge_agent import JudgeAgent, JudgeResponse

__all__ = [
    "AgentResponse",
    "BaseAgent",
    "AnalystAgent",
    "SkepticAgent",
    "SkepticResponse",
    "JudgeAgent",
    "JudgeResponse",
]
