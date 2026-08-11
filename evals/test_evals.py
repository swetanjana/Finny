"""Pytest wrapper around evals/cases.py so `uv run pytest evals/` works as the
Phase 7 gate, in addition to `uv run python -m evals.run_evals`. Deterministic —
no GROQ_API_KEY or network call required."""

import pytest

from evals.cases import CASES


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_case(case):
    case.run()
