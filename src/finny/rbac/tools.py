"""RBAC-enforced data access tools: metric lookup and narrative search.

Enforcement happens inside the query itself (SQL `WHERE sensitivity IN (...)`,
Chroma `where={"sensitivity": {"$in": [...]}}`)  — restricted rows/chunks are
never fetched for a role lacking the tag, so there is nothing to discard after
the fact. Every call is logged to the audit table with role, allowed tags, and
what was actually retrieved (or why it was denied).
"""

import sqlite3
from typing import Any, Dict, List, Optional

from finny.feedback import store as feedback_store
from finny.paths import CHROMA_DIR, DB_PATH
from finny.rbac.context import PermissionContext
from finny.rbac.derived_metrics import DerivedMetricAccessDenied, check_derived_metric_access
from finny.understanding import db as db_module
from finny.understanding import vector_store

# How many extra candidates to over-fetch from Chroma so feedback-based
# re-ranking (Phase 5) has room to reorder before truncating to n_results —
# without this, a downvoted top hit could only ever be dropped, never demoted
# below something currently ranked lower.
RERANK_OVERFETCH_FACTOR = 3


def _connect() -> sqlite3.Connection:
    return db_module.get_connection(DB_PATH)


def _restricted_sensitivities_for(
    conn: sqlite3.Connection, metric: str, period: Optional[str], allowed: List[str]
) -> Optional[List[str]]:
    """When a metric lookup returns nothing, distinguishes "genuinely doesn't exist"
    from "exists but only under a sensitivity tag this role lacks". Only the
    `sensitivity` column is read here — never `value` — so this cannot leak the
    restricted figures themselves, only the fact that a matching, gated row exists.
    Returns None if no matching row exists at all (any sensitivity)."""
    sql = "SELECT DISTINCT sensitivity FROM metrics WHERE metric LIKE ?"
    params: List[Any] = [f"%{metric}%"]
    if period:
        sql += " AND period = ?"
        params.append(period)
    cur = conn.cursor()
    cur.execute(sql, params)
    all_sensitivities = {r[0] for r in cur.fetchall()}
    if not all_sensitivities:
        return None
    restricted = sorted(all_sensitivities - set(allowed))
    return restricted or None


def _fetch_metric_rows(
    conn: sqlite3.Connection, allowed: List[str], metric: str, period: Optional[str], exact: bool
) -> List[Dict[str, Any]]:
    placeholders = ",".join("?" for _ in allowed)
    comparison = "metric = ? COLLATE NOCASE" if exact else "metric LIKE ?"
    sql = (
        "SELECT id, metric, period, value, unit, source, sensitivity, file, doc_type, fiscal_year "
        f"FROM metrics WHERE {comparison} AND sensitivity IN ({placeholders})"
    )
    params: List[Any] = [metric if exact else f"%{metric}%", *allowed]
    if period:
        sql += " AND period = ?"
        params.append(period)
    cur = conn.cursor()
    cur.execute(sql, params)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def lookup_metric(
    context: PermissionContext,
    metric: str,
    period: Optional[str] = None,
    derived_metric_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Exact metric lookup, RBAC-filtered at the SQL layer.

    If `derived_metric_key` names a computed metric (e.g. "revenue_per_employee"),
    the role's access is checked against that computation's full sensitivity
    requirement BEFORE the query runs, so a role that would only be short one
    input never gets a silently partial result.
    """
    conn = _connect()
    try:
        if derived_metric_key is not None:
            try:
                check_derived_metric_access(derived_metric_key, context)
            except DerivedMetricAccessDenied as exc:
                db_module.log_audit(
                    conn, context.role, context.allowed_sensitivities, "lookup_metric",
                    {"metric": metric, "period": period, "derived_metric_key": derived_metric_key},
                    [], denied_reason=str(exc),
                )
                return {"status": "denied", "reason": str(exc), "results": []}

        allowed = list(context.allowed_sensitivities)

        # Try an exact (case-insensitive) name match first — this is what
        # disambiguates e.g. "Net income" from unrelated rows that merely
        # *contain* it ("in net income" cash-flow adjustment lines) without
        # ever needing a substring fallback. Only when nothing matches the
        # name exactly do we fall back to a fuzzy LIKE, and a fuzzy match that
        # spans more than one distinct underlying metric name is refused as
        # ambiguous rather than silently handing back a mix (e.g. "revenue"
        # LIKE-matching both "Services Revenue" and "Deferred revenue" while
        # missing "Total net sales" — the real total — entirely).
        rows = _fetch_metric_rows(conn, allowed, metric, period, exact=True)
        if not rows:
            rows = _fetch_metric_rows(conn, allowed, metric, period, exact=False)
            if rows:
                distinct = sorted({r["metric"].strip().casefold(): r["metric"] for r in rows}.values())
                if len(distinct) > 1:
                    reason = (
                        f"'{metric}' matches more than one distinct metric name: {distinct}. "
                        "Call lookup_metric again with one of these exact names, or use "
                        "list_available_metrics to see all names this role can view."
                    )
                    db_module.log_audit(
                        conn, context.role, context.allowed_sensitivities, "lookup_metric",
                        {"metric": metric, "period": period}, [r["id"] for r in rows],
                        denied_reason=reason,
                    )
                    return {"status": "ambiguous", "reason": reason, "candidates": distinct, "results": []}

        if rows:
            # Re-ranking (Phase 5): reorder by feedback history on similar past
            # queries — never drops a row, only changes which the model sees first
            # (and therefore tends to cite), since an exact-metric answer shouldn't
            # disappear just because a differently-phrased past query on it was
            # downvoted.
            adjustments = feedback_store.get_relevance_adjustments(conn, metric)
            if adjustments:
                rows.sort(key=lambda r: adjustments.get(r["id"], 0.0), reverse=True)

        if not rows:
            restricted = _restricted_sensitivities_for(conn, metric, period, allowed)
            if restricted:
                reason = (
                    f"Role '{context.role}' does not have access to '{metric}': it is classified "
                    f"under sensitivity tag(s) {restricted}, which this role is not permitted to view."
                )
                db_module.log_audit(
                    conn, context.role, context.allowed_sensitivities, "lookup_metric",
                    {"metric": metric, "period": period}, [], denied_reason=reason,
                )
                return {"status": "restricted", "reason": reason, "results": []}

        db_module.log_audit(
            conn, context.role, context.allowed_sensitivities, "lookup_metric",
            {"metric": metric, "period": period}, [r["id"] for r in rows],
        )
        return {"status": "ok", "results": rows}
    finally:
        conn.close()


def search_filings(
    context: PermissionContext,
    query_text: str,
    n_results: int = 5,
    doc_type: Optional[str] = None,
    fiscal_year: Optional[str] = None,
) -> Dict[str, Any]:
    """Semantic search over narrative chunks, RBAC-filtered inside the Chroma query.

    Every narrative chunk is tagged `public_narrative` — it's the only sensitivity
    the Chroma collection ever holds — so a role lacking that tag is refused up
    front, without even querying Chroma, rather than silently returning zero hits.
    """
    if "public_narrative" not in context.allowed_sensitivities:
        reason = (
            f"Role '{context.role}' does not have access to filing narrative text "
            f"(sensitivity tag 'public_narrative')."
        )
        conn = _connect()
        try:
            db_module.log_audit(
                conn, context.role, context.allowed_sensitivities, "search_filings",
                {"query_text": query_text, "n_results": n_results, "doc_type": doc_type, "fiscal_year": fiscal_year},
                [], denied_reason=reason,
            )
        finally:
            conn.close()
        return {"status": "restricted", "reason": reason, "results": []}

    where_extra: Dict[str, Any] = {}
    if doc_type:
        where_extra["doc_type"] = doc_type
    if fiscal_year:
        where_extra["fiscal_year"] = fiscal_year

    result = vector_store.query(
        CHROMA_DIR, query_text, list(context.allowed_sensitivities),
        n_results=n_results * RERANK_OVERFETCH_FACTOR, where_extra=where_extra or None,
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]

    conn = _connect()
    try:
        # Re-ranking (Phase 5): over-fetched n_results*3 candidates above so a
        # feedback-penalized chunk can actually be demoted below others, not just
        # dropped. Combines Chroma's cosine distance (lower = better, so negated
        # into a score) with the feedback-derived adjustment before truncating to
        # the caller's requested n_results.
        adjustments = feedback_store.get_relevance_adjustments(conn, query_text)
        ranked = sorted(
            zip(ids, documents, metadatas, distances),
            key=lambda item: -item[3] + adjustments.get(item[0], 0.0),
            reverse=True,
        )[:n_results]
        ids = [r[0] for r in ranked]
        documents = [r[1] for r in ranked]
        metadatas = [r[2] for r in ranked]

        db_module.log_audit(
            conn, context.role, context.allowed_sensitivities, "search_filings",
            {"query_text": query_text, "n_results": n_results, "doc_type": doc_type, "fiscal_year": fiscal_year},
            ids,
        )
    finally:
        conn.close()

    return {
        "status": "ok",
        "results": [{"id": i, "content": d, **m} for i, d, m in zip(ids, documents, metadatas)],
    }


def list_available_metrics(context: PermissionContext) -> Dict[str, Any]:
    """Distinct metric names visible to this role — lets an agent/user discover what
    it can ask about without leaking names of metrics it can't see."""
    conn = _connect()
    try:
        allowed = list(context.allowed_sensitivities)
        placeholders = ",".join("?" for _ in allowed)
        cur = conn.cursor()
        cur.execute(
            f"SELECT DISTINCT metric FROM metrics WHERE sensitivity IN ({placeholders}) ORDER BY metric",
            allowed,
        )
        metrics = [r[0] for r in cur.fetchall()]
        db_module.log_audit(
            conn, context.role, context.allowed_sensitivities, "list_available_metrics", {}, metrics,
        )
        return {"status": "ok", "results": metrics}
    finally:
        conn.close()
