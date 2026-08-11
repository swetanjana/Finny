"""FastAPI route handlers for Finny web API.

Four endpoints:
  POST /api/ask        — SSE streaming agent response
  POST /api/feedback   — record thumbs-up/down or correction
  GET  /api/metrics    — list metrics visible to a given role (no LLM needed)
  GET  /api/status     — ingestion dataset statistics
"""

import json
import os
import sqlite3
from typing import Generator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from finny.agent.loop import stream_agent, _extract_tool_results
from finny.feedback import store as feedback_store
from finny.paths import CHROMA_DIR, DB_PATH
from finny.rbac import tools as rbac_tools
from finny.rbac.context import PermissionContext
from finny.rbac.roles import UnknownRoleError
from finny.understanding import db as db_module
from finny.interface.web.schemas import AskRequest, FeedbackRequest

router = APIRouter(prefix="/api")


def _get_conn() -> sqlite3.Connection:
    return db_module.get_connection(DB_PATH)


def _sse_event(data: dict) -> str:
    """Format one SSE frame."""
    return f"data: {json.dumps(data, default=str)}\n\n"


def _resolve_context(role: str) -> PermissionContext:
    try:
        return PermissionContext(role)
    except UnknownRoleError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/ask")
def ask(body: AskRequest) -> StreamingResponse:
    """SSE stream: one JSON frame per LangGraph super-step plus a final 'done' frame
    carrying the sources table and query_id for feedback.

    Plain (sync) def — FastAPI runs this in a threadpool automatically, which is the
    correct way to use blocking Groq HTTP calls without blocking the event loop.
    """
    context = _resolve_context(body.role)
    api_key = os.environ.get("GROQ_API_KEY")

    def generate() -> Generator[str, None, None]:
        tool_results = []
        final_answer = ""

        for event in stream_agent(context, body.query, api_key=api_key):
            if event["type"] == "final_answer":
                final_answer = event["answer"]
            elif event["type"] == "done":
                tool_results = event.get("tool_results", [])
                # Record to query_log and inject query_id into the done frame
                retrieved_ids = [
                    rid
                    for tr in tool_results
                    for rid in (tr["result"].get("results") or [])
                    if isinstance(rid, dict) and rid.get("id")
                ]
                # Flatten list-of-dicts ids to list-of-strings
                flat_ids = []
                for tr in tool_results:
                    res = tr["result"]
                    for r in res.get("results") or []:
                        if isinstance(r, dict) and r.get("id"):
                            flat_ids.append(r["id"])
                        elif isinstance(r, str):
                            flat_ids.append(r)

                query_id = None
                try:
                    conn = _get_conn()
                    query_id = feedback_store.record_query(
                        conn, body.role, body.query, final_answer, flat_ids
                    )
                    conn.close()
                except Exception:
                    pass  # non-critical; don't crash the stream on logging failure

                done_event = {**event, "query_id": query_id}
                yield _sse_event(done_event)
                return  # already yielded done

            yield _sse_event(event)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/feedback")
def feedback(body: FeedbackRequest):
    """Record user feedback (thumbs-up/down or correction text) for a past query."""
    if body.rating is None and not body.correction_text:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: rating ('up' or 'down'), correction_text.",
        )

    # Mirror CLI behaviour: if only correction_text is given, default rating = "down"
    rating = body.rating or "down"

    try:
        conn = _get_conn()
        feedback_id = feedback_store.record_feedback(
            conn, CHROMA_DIR, body.query_id, rating, correction_text=body.correction_text
        )
        conn.close()
    except feedback_store.FeedbackTargetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {"status": "ok", "feedback_id": feedback_id}


@router.get("/metrics")
def metrics(role: str):
    """Distinct metric names visible to the given role. No LLM call needed — pure
    RBAC-filtered SQL query. Used by the sidebar to refresh on role switch."""
    context = _resolve_context(role)
    result = rbac_tools.list_available_metrics(context)
    return result


@router.get("/status")
def status():
    """Ingestion dataset statistics (total files, records, sensitivity breakdown)."""
    from finny.paths import PROCESSED_DIR
    summary_path = PROCESSED_DIR / "ingestion_summary.json"
    if not summary_path.exists():
        return {"status": "no_data", "message": "Run 'finny build-index' first."}
    with open(summary_path) as f:
        return json.load(f)
