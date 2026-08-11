#!/usr/bin/env python
"""Phase 7 eval gate.

Runs the deterministic RBAC + metric-accuracy suite (evals/cases.py) against
the tool layer directly — no GROQ_API_KEY or network call required. Pass
--agent to additionally run the smaller end-to-end NL suite (evals/agent_cases.py)
through the real LangGraph + Groq agent loop.

Usage:
    uv run python -m evals.run_evals            # deterministic gate
    uv run python -m evals.run_evals --agent     # + live end-to-end NL suite

Also runnable via pytest: `uv run pytest evals/` (see evals/test_evals.py and
evals/test_agent_evals.py).
"""

import argparse
import sys
from pathlib import Path
from typing import Callable, Tuple

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from finny.paths import DB_PATH, CHROMA_DIR

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

console = Console()


def _run_case(run: Callable[[], None]) -> Tuple[bool, str]:
    try:
        run()
        return True, ""
    except AssertionError as exc:
        return False, str(exc)
    except Exception as exc:  # a case erroring out is still a failure, not a crash of the whole gate
        return False, f"{type(exc).__name__}: {exc}"


def _run_suite(title: str, cases) -> list:
    table = Table(title=title)
    table.add_column("ID", style="cyan")
    table.add_column("Description")
    table.add_column("Result")

    failures = []
    for case in cases:
        ok, message = _run_case(case.run)
        table.add_row(case.id, case.description, "[green]PASS[/green]" if ok else f"[red]FAIL[/red]: {message}")
        if not ok:
            failures.append(case.id)

    console.print(table)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Finny eval gate")
    parser.add_argument(
        "--agent",
        action="store_true",
        help="also run the live end-to-end NL agent suite (requires GROQ_API_KEY, makes real API calls)",
    )
    args = parser.parse_args()

    if not DB_PATH.exists() or not CHROMA_DIR.exists():
        console.print(
            "[red]data/db/finny.sqlite or data/index/chroma is missing — run "
            "'uv run finny build-index' first (or './scripts/setup.sh' for a fresh clone).[/red]"
        )
        return 1

    from evals.cases import CASES

    failures = _run_suite("Finny eval gate — RBAC + metric accuracy (deterministic, no LLM)", CASES)
    total = len(CASES)

    if args.agent:
        from evals.agent_cases import AGENT_CASES

        failures += _run_suite("Finny eval gate — end-to-end NL agent suite (live Groq calls)", AGENT_CASES)
        total += len(AGENT_CASES)

    if failures:
        console.print(f"[bold red]{len(failures)}/{total} case(s) failed: {failures}[/bold red]")
        return 1

    console.print(f"[bold green]All {total} case(s) passed.[/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
