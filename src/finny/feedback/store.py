"""Phase 5 — feedback logging and its two effects on future retrieval:

1. Re-ranking: `get_relevance_adjustments()` turns feedback history into a
   {id: score_delta} map — thumbs-down on a chunk/metric penalizes it, thumbs-up
   boosts it, weighted by how similar the past feedback query is to the current
   one (embedding cosine similarity, same local sentence-transformers model used
   for narrative search — no second model load).
2. Few-shot memory: upvoted or corrected (query, answer) pairs are embedded and
   stored in a small Chroma collection (`feedback_examples`); `get_fewshot_examples()`
   retrieves the most similar past ones for injection into the agent's prompt.

Both read/write through `finny.understanding.db` (SQLite: query_log, feedback)
and `finny.understanding.vector_store` (Chroma client + embedding function) rather
than opening their own connections/models, so there is exactly one cached
embedding-model instance and one Chroma client per process.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from finny.understanding import db as db_module
from finny.understanding.vector_store import get_client, get_embedding_function

FEWSHOT_COLLECTION = "feedback_examples"
VALID_RATINGS = {"up", "down"}

# How similar a past feedback query must be (cosine similarity, 0..1) to the
# current query before its rating affects re-ranking — keeps a downvote on
# "headcount" from bleeding into an unrelated "R&D spend" search.
RELEVANCE_SIMILARITY_THRESHOLD = 0.45


class FeedbackTargetNotFound(Exception):
    pass


def record_query(conn: sqlite3.Connection, role: str, query: str, answer: str, retrieved_ids: List[str]) -> int:
    return db_module.log_query(conn, role, query, answer, retrieved_ids)


def record_feedback(
    conn: sqlite3.Connection,
    chroma_dir: Path,
    query_id: int,
    rating: str,
    correction_text: Optional[str] = None,
) -> int:
    """Writes one feedback row tied to a previously logged query, and — if the
    feedback carries a "good" answer (an upvote, or any correction) — adds it to
    the few-shot example store. A bare downvote has no good answer to show, so it
    only feeds re-ranking, never few-shot memory."""
    if rating not in VALID_RATINGS:
        raise ValueError(f"rating must be one of {sorted(VALID_RATINGS)}, got {rating!r}")

    logged = db_module.get_query(conn, query_id)
    if logged is None:
        raise FeedbackTargetNotFound(
            f"No logged query with id {query_id}. Query ids are printed after each 'finny ask'/'finny chat' answer."
        )

    feedback_id = db_module.log_feedback(
        conn,
        query_id=query_id,
        query=logged["query"],
        role=logged["role"],
        retrieved_ids=logged["retrieved_ids"],
        answer=logged["answer"],
        rating=rating,
        correction_text=correction_text,
    )

    if rating == "up" or correction_text:
        example_answer = correction_text or logged["answer"]
        _add_fewshot_example(chroma_dir, feedback_id, logged["role"], logged["query"], example_answer, rating)

    return feedback_id


def _cosine_similarities(query_vec: List[float], candidate_vecs: List[List[float]]) -> np.ndarray:
    q = np.asarray(query_vec, dtype=np.float32)
    c = np.asarray(candidate_vecs, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-9)
    c_norm = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
    return c_norm @ q_norm


def get_relevance_adjustments(conn: sqlite3.Connection, query_text: str) -> Dict[str, float]:
    """{id: score_delta} built from feedback on queries similar to `query_text`.
    Computed fresh per call rather than cached/precomputed — feedback volume at
    demo scale (tens to low hundreds of rows) makes this cheap, and it means a
    feedback event affects the very next query with no rebuild step."""
    rows = db_module.get_all_feedback(conn)
    if not rows:
        return {}

    embed_fn = get_embedding_function()
    vectors = embed_fn([query_text] + [r["query"] for r in rows])
    sims = _cosine_similarities(vectors[0], vectors[1:])

    adjustments: Dict[str, float] = {}
    for sim, row in zip(sims, rows):
        if sim < RELEVANCE_SIMILARITY_THRESHOLD:
            continue
        weight = float(sim) if row["rating"] == "up" else -float(sim)
        for item_id in row["retrieved_ids"]:
            adjustments[item_id] = adjustments.get(item_id, 0.0) + weight
    return adjustments


def _fewshot_collection(chroma_dir: Path):
    client = get_client(chroma_dir)
    return client.get_or_create_collection(
        FEWSHOT_COLLECTION,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def _add_fewshot_example(
    chroma_dir: Path, feedback_id: int, role: str, query_text: str, example_answer: str, rating: str
) -> None:
    collection = _fewshot_collection(chroma_dir)
    collection.add(
        ids=[f"feedback_{feedback_id}"],
        documents=[query_text],
        metadatas=[{"role": role, "answer": example_answer, "rating": rating}],
    )


def get_fewshot_examples(chroma_dir: Path, role: str, query_text: str, k: int = 2) -> List[Dict[str, str]]:
    """Top-k past (query, answer) pairs most similar to `query_text`, scoped to
    `role`. The role filter is a hard RBAC boundary, not just relevance tuning:
    an example generated for `ceo` may contain headcount/compensation figures,
    and injecting it into an `analyst` or `cto` prompt would leak restricted data
    through the few-shot channel even though the live tool calls stay filtered.

    Chroma's nearest-neighbor query always returns k results if k exist, even if
    none are actually relevant — without a floor, every turn pays for injecting
    two unrelated examples' worth of tokens into the Groq call. Filtered by the
    same RELEVANCE_SIMILARITY_THRESHOLD used for re-ranking, so an example is only
    injected when it's similar enough to matter."""
    collection = _fewshot_collection(chroma_dir)
    if collection.count() == 0:
        return []
    result = collection.query(
        query_texts=[query_text],
        n_results=min(k, collection.count()),
        where={"role": role},
        include=["documents", "metadatas", "distances"],
    )
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return [
        {"query": d, "answer": m.get("answer", "")}
        for d, m, dist in zip(documents, metadatas, distances)
        if (1 - dist) >= RELEVANCE_SIMILARITY_THRESHOLD
    ]
