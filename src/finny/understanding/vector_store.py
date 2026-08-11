"""Chroma vector store for narrative chunks, embedded locally with sentence-transformers."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

import chromadb
from chromadb.utils import embedding_functions

COLLECTION_NAME = "narrative_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_function():
    # Local, offline sentence-transformers model — no external embedding API.
    # Cached: loading the model from disk takes ~1s+, and every tool call /
    # feedback-similarity check would otherwise reload it from scratch.
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=4)
def get_client(persist_dir: Path) -> chromadb.ClientAPI:
    # Cached per persist_dir: PersistentClient() re-opens the on-disk sqlite index
    # on every construction, which is unnecessary overhead when called once per
    # tool invocation (or once per turn, in a multi-turn chat session).
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_index(persist_dir: Path, narrative_jsonl_path: Path, batch_size: int = 256) -> int:
    """(Re)builds the narrative_chunks Chroma collection from narrative_chunks.jsonl.

    Idempotent: drops and recreates the collection so re-running scripts/build_index.py
    always reflects the current contents of data/raw/ rather than accumulating duplicates.
    """
    client = get_client(persist_dir)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for rec in _iter_jsonl(narrative_jsonl_path):
        ids.append(rec["id"])
        documents.append(rec["content"])
        metadatas.append(
            {
                "sensitivity": rec["sensitivity"],
                "doc_type": rec.get("doc_type", ""),
                "fiscal_year": rec.get("fiscal_year", ""),
                "section": rec.get("section", ""),
                "source_page": rec.get("source_page", 0),
                "source": rec.get("file", ""),
                # Phase 6: instruction-like patterns flagged at ingestion time.
                # Chroma metadata values must be scalars, so the reason list is
                # joined into a comma-separated string rather than stored as-is.
                "flagged": bool(rec.get("flagged", False)),
                "flag_reasons": ",".join(rec.get("flag_reasons", [])),
            }
        )

    total = len(ids)
    for start in range(0, total, batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    return total


def query(
    persist_dir: Path,
    query_text: str,
    allowed_sensitivities: List[str],
    n_results: int = 5,
    where_extra: Dict[str, Any] | None = None,
):
    """RBAC-filtered similarity search: sensitivity filter is applied inside the Chroma
    query itself (server-side `where`), not post-filtered after retrieval — restricted
    chunks are never returned to a caller lacking the tag."""
    client = get_client(persist_dir)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=get_embedding_function())

    where: Dict[str, Any] = {"sensitivity": {"$in": allowed_sensitivities}}
    if where_extra:
        where = {"$and": [where, where_extra]}

    return collection.query(query_texts=[query_text], n_results=n_results, where=where)
