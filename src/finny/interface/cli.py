"""Typer CLI interface for Finny agent system."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from finny.feedback import store as feedback_store
from finny.ingestion.pipeline import IngestionPipeline
from finny.understanding.build_index import run as run_build_index
from finny.understanding import db as db_module
from finny.agent.loop import run_agent
from finny.paths import CHROMA_DIR, DB_PATH
from finny.rbac.context import PermissionContext
from finny.rbac.roles import UnknownRoleError

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

app = typer.Typer(name="finny", help="Finny — Financial-Data Agent CLI")
console = Console()


def _extract_retrieved_ids(tool_results: List[Dict[str, Any]]) -> List[str]:
    """Flattens every id retrieved across a turn's tool calls (search_filings hits,
    lookup_metric rows) — this is what feedback on the turn applies to."""
    ids: List[str] = []
    for tr in tool_results:
        results = tr["result"].get("results")
        if isinstance(results, list):
            ids.extend(r["id"] for r in results if isinstance(r, dict) and "id" in r)
    return ids


def _render_answer(role: str, query: str, result: dict) -> Optional[int]:
    console.print(Panel(result["answer"], title=f"Finny ({role})", subtitle=query, border_style="blue"))

    tool_results = result.get("tool_results", [])
    if tool_results:
        table = Table(title="Tool calls / sources")
        table.add_column("Tool", style="cyan")
        table.add_column("Args")
        table.add_column("Status")
        table.add_column("Retrieved")

        for tr in tool_results:
            res = tr["result"]
            status = res.get("status", "?")
            if isinstance(res.get("results"), list):
                retrieved = str(len(res["results"]))
            elif "price" in res:
                retrieved = "1"
            else:
                retrieved = "0"
            table.add_row(tr["tool"], json.dumps(tr["arguments"]), status, retrieved)

        console.print(table)

    # Phase 5: log this turn so `finny feedback <query_id>` can reference it.
    conn = db_module.get_connection(DB_PATH)
    try:
        query_id = feedback_store.record_query(conn, role, query, result["answer"], _extract_retrieved_ids(tool_results))
    finally:
        conn.close()

    console.print(f"[dim]Query ID: {query_id} — give feedback with 'finny feedback {query_id} --up/--down/--correct \"...\"'[/dim]")
    return query_id


def _make_context(role: str) -> PermissionContext:
    try:
        return PermissionContext(role)
    except UnknownRoleError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)


@app.command()
def ingest(
    raw_dir: Path = typer.Option(None, "--raw-dir", help="Path to raw input directory"),
    processed_dir: Path = typer.Option(None, "--processed-dir", help="Path to processed output directory"),
):
    """Run Phase 1 data ingestion over raw SEC filings and curated financial workbooks."""
    root_dir = Path(__file__).resolve().parents[3]
    if raw_dir is None:
        raw_dir = root_dir / "data" / "raw"
    if processed_dir is None:
        processed_dir = root_dir / "data" / "processed"

    pipeline = IngestionPipeline(raw_dir, processed_dir)
    pipeline.run()


@app.command(name="build-index")
def build_index():
    """Regenerate everything derived from data/raw/: ingestion (Phase 1) plus the
    SQLite metrics table, Chroma narrative index, and per-filing understanding
    files (Phase 2). Idempotent — safe to re-run after raw data changes."""
    root_dir = Path(__file__).resolve().parents[3]
    run_build_index(root_dir)


@app.command()
def ask(
    query: str = typer.Argument(..., help="Natural-language question to ask Finny"),
    role: str = typer.Option(..., "--role", help="Caller role: ceo | cto | analyst"),
):
    """Ask Finny a single question as a given role. RBAC is enforced at the data layer."""
    context = _make_context(role)
    result = run_agent(context, query)
    _render_answer(role, query, result)


@app.command()
def chat(
    role: str = typer.Option(..., "--role", help="Caller role: ceo | cto | analyst"),
):
    """Interactive REPL: ask Finny multiple questions as a given role. Type 'exit' to quit."""
    context = _make_context(role)
    console.print(f"[bold blue]Finny chat[/bold blue] — role=[cyan]{role}[/cyan]. Type 'exit' or 'quit' to leave.")

    # Phase 4.5: process-local conversation memory for this session — not
    # persisted, just accumulated in RAM and passed forward each turn so
    # follow-up questions ("and what about the year before that?") resolve.
    history: List[Any] = []

    while True:
        try:
            query = typer.prompt(">")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if query.strip().lower() in {"exit", "quit"}:
            break
        if not query.strip():
            continue

        result = run_agent(context, query, history=history)
        _render_answer(role, query, result)

        history.append(HumanMessage(content=f"[Caller role: {role}] {query}"))
        history.append(AIMessage(content=result["answer"]))


@app.command()
def feedback(
    query_id: int = typer.Argument(..., help="Query ID printed after a 'finny ask'/'finny chat' answer"),
    up: bool = typer.Option(False, "--up", help="Thumbs-up: the answer was correct"),
    down: bool = typer.Option(False, "--down", help="Thumbs-down: the answer was wrong or incomplete"),
    correct: Optional[str] = typer.Option(
        None, "--correct", help="The correct answer text — implies --down unless --up is also given"
    ),
):
    """Rate a past answer. Feeds two things back into future queries:
    re-ranking (thumbs-down penalizes, thumbs-up boosts the same chunks/metrics
    for similar future queries) and few-shot memory (upvoted or corrected answers
    are shown as examples to the agent on similar future queries, scoped to the
    same role)."""
    if up and down:
        console.print("[red]Pass at most one of --up / --down.[/red]")
        raise typer.Exit(code=1)
    if not up and not down and not correct:
        console.print("[red]Pass --up, --down, and/or --correct \"...\".[/red]")
        raise typer.Exit(code=1)

    rating = "up" if up else "down"

    conn = db_module.get_connection(DB_PATH)
    try:
        feedback_id = feedback_store.record_feedback(conn, CHROMA_DIR, query_id, rating, correction_text=correct)
    except feedback_store.FeedbackTargetNotFound as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    finally:
        conn.close()

    console.print(f"[green]Recorded feedback #{feedback_id} ({rating}) for query {query_id}.[/green]")


@app.command()
def status():
    """Show system status and dataset statistics."""
    root_dir = Path(__file__).resolve().parents[3]
    processed_dir = root_dir / "data" / "processed"
    summary_file = processed_dir / "ingestion_summary.json"

    if not summary_file.exists():
        console.print("[yellow]Ingestion has not been run yet. Run 'finny ingest' first.[/yellow]")
        return

    import json
    with open(summary_file, "r") as f:
        data = json.load(f)

    console.print("[bold green]Finny Status: Ingestion Phase Complete[/bold green]")
    console.print(f"Total Files Processed: {data.get('total_files_processed')}")
    console.print(f"Financial Metrics Records: {data.get('total_financial_metrics')}")
    console.print(f"Narrative Chunks Extracted: {data.get('total_narrative_chunks')}")
    console.print("Sensitivity Breakdown:")
    for k, v in data.get("sensitivity_breakdown", {}).items():
        console.print(f"  - [cyan]{k}[/cyan]: {v}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host (0.0.0.0 to expose on LAN)"),
    port: int = typer.Option(8000, "--port", help="Port to listen on"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on source changes (dev only)"),
):
    """Run the Finny web UI (FastAPI + SSE streaming). Opens at http://<host>:<port>/

    This is a long-lived process — unlike each CLI command which exits after one
    answer, the server reuses the cached LangGraph graph and embedding model across
    all requests, so the second request in a session is noticeably faster than the
    first (no graph-compile or model-load overhead)."""
    import uvicorn
    console.print(f"[bold blue]Finny web UI[/bold blue] → http://{host}:{port}/")
    console.print("[dim]Press Ctrl+C to stop.[/dim]")
    uvicorn.run(
        "finny.interface.web.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
