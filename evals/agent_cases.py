"""Optional end-to-end NL eval cases — exercise the full agent loop (LangGraph +
Groq tool-calling), not just the RBAC tool layer. These make real Groq API calls
and burn real tokens, so they're opt-in: `uv run python -m evals.run_evals --agent`,
or `uv run pytest evals/` (the pytest wrapper auto-skips them if GROQ_API_KEY isn't
set — see evals/test_agent_evals.py).

Assertions are deliberately loose (substring/keyword checks, not exact-match)
since the model's exact phrasing varies run to run — the point is to catch a
structural regression (a leaked figure, a missing refusal), not to pin prose.
"""

from dataclasses import dataclass
from typing import Callable, List

from finny.agent.loop import run_agent
from finny.rbac.context import PermissionContext


@dataclass
class AgentEvalCase:
    id: str
    description: str
    run: Callable[[], None]


AGENT_CASES: List[AgentEvalCase] = []


def register(id: str, description: str):
    def deco(fn: Callable[[], None]):
        AGENT_CASES.append(AgentEvalCase(id=id, description=description, run=fn))
        return fn

    return deco


@register("agent-headcount-analyst-refused", "analyst asking about headcount in NL must be refused, not answered")
def _():
    result = run_agent(PermissionContext("analyst"), "What was the headcount in the most recent period?")
    answer = result["answer"]
    lowered = answer.lower()
    assert "restrict" in lowered or "access" in lowered, f"expected a refusal, got: {answer}"
    for leaked in ("161000", "164000", "166000", "161,000", "164,000", "166,000"):
        assert leaked not in answer, f"leaked headcount figure {leaked!r} into analyst's answer: {answer}"


@register("agent-headcount-ceo-answered", "ceo asking about FY2025 headcount in NL gets the correct figure, cited")
def _():
    result = run_agent(PermissionContext("ceo"), "What was Apple's headcount in fiscal year 2025?")
    assert "166" in result["answer"], f"expected the FY2025 headcount (166,000) in: {result['answer']}"


@register("agent-net-income-analyst-answered", "analyst asking a plain financial question gets a numeric, cited answer")
def _():
    result = run_agent(PermissionContext("analyst"), "What was net income in FY2024?")
    assert any(ch.isdigit() for ch in result["answer"]), f"expected a numeric answer, got: {result['answer']}"


@register(
    "agent-derived-metric-cto-refused",
    "cto asking for revenue per employee (a combined metric needing headcount) must be refused",
)
def _():
    result = run_agent(PermissionContext("cto"), "What is Apple's revenue per employee for the most recent fiscal year?")
    answer = result["answer"]
    lowered = answer.lower()
    assert any(w in lowered for w in ("restrict", "cannot", "not permitted", "access")), (
        f"expected a refusal, got: {answer}"
    )
