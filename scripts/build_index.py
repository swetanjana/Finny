#!/usr/bin/env python
"""Single idempotent entrypoint: regenerates Phase 1 (ingestion) + Phase 2
(SQLite metrics, Chroma narrative index, understanding files) from data/raw/.

Usage: uv run python scripts/build_index.py
"""

from pathlib import Path

from finny.understanding.build_index import run

if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    run(root_dir)
