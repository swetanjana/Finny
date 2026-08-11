"""SQLite loader for structured financial metrics (Phase 2 understanding layer)
and the RBAC access audit log (Phase 3)."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS metrics (
    id TEXT PRIMARY KEY,
    metric TEXT NOT NULL,
    period TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    source TEXT,
    sensitivity TEXT NOT NULL,
    file TEXT,
    sheet TEXT,
    doc_type TEXT,
    fiscal_year TEXT
);

CREATE INDEX IF NOT EXISTS idx_metrics_lookup ON metrics (metric, period, sensitivity);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    role TEXT NOT NULL,
    allowed_sensitivities TEXT NOT NULL,
    tool TEXT NOT NULL,
    query_params TEXT,
    retrieved_count INTEGER NOT NULL,
    retrieved_ids TEXT,
    denied_reason TEXT
);

CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    role TEXT NOT NULL,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    retrieved_ids TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id INTEGER NOT NULL REFERENCES query_log(id),
    timestamp TEXT NOT NULL,
    query TEXT NOT NULL,
    role TEXT NOT NULL,
    retrieved_ids TEXT NOT NULL,
    answer TEXT NOT NULL,
    rating TEXT NOT NULL,
    correction_text TEXT
);
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_metrics(conn: sqlite3.Connection, metrics_jsonl_path: Path) -> int:
    """Loads financial_metrics.jsonl into the metrics table. Idempotent: clears table first
    so re-running scripts/build_index.py always reflects the current contents of data/raw/."""
    cur = conn.cursor()
    cur.execute("DELETE FROM metrics")

    rows = [
        (
            rec["id"],
            rec["metric"],
            rec["period"],
            rec["value"],
            rec.get("unit"),
            rec.get("source"),
            rec["sensitivity"],
            rec.get("file"),
            rec.get("sheet"),
            rec.get("doc_type"),
            rec.get("fiscal_year"),
        )
        for rec in _iter_jsonl(metrics_jsonl_path)
    ]

    cur.executemany(
        """INSERT OR REPLACE INTO metrics
           (id, metric, period, value, unit, source, sensitivity, file, sheet, doc_type, fiscal_year)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return len(rows)


def log_query(
    conn: sqlite3.Connection,
    role: str,
    query: str,
    answer: str,
    retrieved_ids: List[str],
) -> int:
    """Logs one top-level agent turn (question + final answer + every id retrieved
    across its tool calls) so a later `finny feedback <query_id>` can reference it.
    Never cleared by build_index — a running log across sessions, like audit_log."""
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO query_log (timestamp, role, query, answer, retrieved_ids) VALUES (?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), role, query, answer, json.dumps(retrieved_ids)),
    )
    conn.commit()
    return cur.lastrowid


def get_query(conn: sqlite3.Connection, query_id: int) -> Optional[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute("SELECT id, role, query, answer, retrieved_ids FROM query_log WHERE id = ?", (query_id,))
    row = cur.fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "role": row[1],
        "query": row[2],
        "answer": row[3],
        "retrieved_ids": json.loads(row[4]),
    }


def log_feedback(
    conn: sqlite3.Connection,
    query_id: int,
    query: str,
    role: str,
    retrieved_ids: List[str],
    answer: str,
    rating: str,
    correction_text: Optional[str] = None,
) -> int:
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO feedback
           (query_id, timestamp, query, role, retrieved_ids, answer, rating, correction_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            query_id,
            datetime.now(timezone.utc).isoformat(),
            query,
            role,
            json.dumps(retrieved_ids),
            answer,
            rating,
            correction_text,
        ),
    )
    conn.commit()
    return cur.lastrowid


def get_all_feedback(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """All feedback rows, for re-ranking score computation. Small table at this
    scale (demo-sized feedback volume) — read in full rather than paginated."""
    cur = conn.cursor()
    cur.execute("SELECT query, retrieved_ids, rating FROM feedback")
    return [
        {"query": r[0], "retrieved_ids": json.loads(r[1]), "rating": r[2]}
        for r in cur.fetchall()
    ]


def log_audit(
    conn: sqlite3.Connection,
    role: str,
    allowed_sensitivities: Iterable[str],
    tool: str,
    query_params: Dict[str, Any],
    retrieved_ids: List[str],
    denied_reason: Optional[str] = None,
) -> None:
    """Appends one row per tool call: who asked, what they were allowed to see, and
    what was actually retrieved. Never cleared by build_index — this is a running
    log across sessions, not derived data."""
    conn.execute(
        """INSERT INTO audit_log
           (timestamp, role, allowed_sensitivities, tool, query_params, retrieved_count, retrieved_ids, denied_reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            role,
            json.dumps(sorted(allowed_sensitivities)),
            tool,
            json.dumps(query_params, default=str),
            len(retrieved_ids),
            json.dumps(retrieved_ids),
            denied_reason,
        ),
    )
    conn.commit()
