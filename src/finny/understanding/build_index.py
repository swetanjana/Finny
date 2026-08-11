"""Phase 1 + Phase 2 orchestrator: regenerates everything derived from data/raw/.

Idempotent — safe to re-run any time raw filings or the curated workbook change.
This is the single entrypoint used by both `scripts/build_index.py` and the
`finny build-index` CLI command.
"""

import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from rich.console import Console

from finny.ingestion.pipeline import IngestionPipeline
from finny.understanding import db, vector_store, summarizer

console = Console()


def run(root_dir: Path) -> Dict[str, Any]:
    load_dotenv(root_dir / ".env")

    raw_dir = root_dir / "data" / "raw"
    processed_dir = root_dir / "data" / "processed"
    db_path = root_dir / "data" / "db" / "finny.sqlite"
    chroma_dir = root_dir / "data" / "index" / "chroma"
    summaries_dir = processed_dir / "summaries"

    # Phase 1 — re-parse raw PDFs/Excel into normalized JSONL.
    console.rule("[bold blue]Phase 1: Ingestion[/bold blue]")
    pipeline = IngestionPipeline(raw_dir, processed_dir)
    ingestion_summary = pipeline.run()

    # Phase 2a — structured metrics -> SQLite.
    console.rule("[bold blue]Phase 2: Understanding Layer[/bold blue]")
    conn = db.get_connection(db_path)
    metrics_loaded = db.load_metrics(conn, processed_dir / "financial_metrics.jsonl")
    conn.close()
    console.print(f"[green]Loaded {metrics_loaded} metric rows into SQLite[/green] ({db_path})")

    # Phase 2b — narrative chunks -> Chroma (local sentence-transformers embeddings).
    chunks_indexed = vector_store.build_index(chroma_dir, processed_dir / "narrative_chunks.jsonl")
    console.print(f"[green]Indexed {chunks_indexed} narrative chunks into Chroma[/green] ({chroma_dir})")

    # Phase 2c — per-filing understanding files via Groq.
    manifest_by_file = {item["file"]: item for item in pipeline.load_manifest()}
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        console.print("[yellow]Warning: GROQ_API_KEY not set — understanding files will skip LLM overviews.[/yellow]")
    written = summarizer.generate_summaries(processed_dir, summaries_dir, manifest_by_file, api_key)
    console.print(f"[green]Wrote {len(written)} understanding files[/green] ({summaries_dir})")

    return {
        "ingestion": ingestion_summary,
        "metrics_loaded": metrics_loaded,
        "narrative_chunks_indexed": chunks_indexed,
        "understanding_files": written,
    }


def main():
    root_dir = Path(__file__).resolve().parents[3]
    run(root_dir)


if __name__ == "__main__":
    main()
