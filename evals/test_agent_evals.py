"""Pytest wrapper around evals/agent_cases.py — the end-to-end NL suite that runs
the real LangGraph + Groq agent loop. Skipped by default (even if GROQ_API_KEY is
set — a key in .env just for interactive 'finny ask' use shouldn't make a plain
`pytest evals/` run burn tokens) unless FINNY_RUN_AGENT_EVALS=1 is set. Equivalent
to `uv run python -m evals.run_evals --agent`."""

import os

import pytest

from evals.agent_cases import AGENT_CASES

pytestmark = pytest.mark.skipif(
    not os.environ.get("FINNY_RUN_AGENT_EVALS"),
    reason="set FINNY_RUN_AGENT_EVALS=1 to run the live-Groq end-to-end suite",
)


@pytest.mark.parametrize("case", AGENT_CASES, ids=[c.id for c in AGENT_CASES])
def test_agent_case(case):
    case.run()
