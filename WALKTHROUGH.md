# Finny — Walkthrough

This is the companion document for the assignment's walkthrough deliverable — what was built, how the pieces fit together, what I'd do differently with more time, and where it breaks at scale. For setup/usage, see [README.md](README.md). (`Explainaibility.md` is a separate, much longer running build log kept for my own reference while developing — this document is the curated version meant to be read or presented on its own.)

## What I built

An agent that answers natural-language questions about Apple's public SEC filings (10-K FY23/24/25, 10-Q FY26-Q3, plus a curated metrics workbook), with three things layered on top of "just retrieval":

```
data/raw/ (PDF + Excel)
      │
      ▼
 ingestion            section-aware PDF chunking, row-level Excel metrics,
                       every record tagged with sensitivity + full provenance
      │
      ▼
 understanding         SQLite (exact metric lookup) + Chroma (narrative
                       semantic search, local embeddings) + one Groq-written
                       summary per filing
      │
      ▼
 RBAC                  role → allowed sensitivity tags, enforced inside the
                       SQL/Chroma query itself — not a post-filter
      │
      ▼
 agent                 LangGraph tool-calling loop over Groq (lookup_metric,
                       search_filings, list_available_metrics, a live stock
                       price tool), multi-turn chat memory
      │
      ▼
 feedback              thumbs-up/down/correction re-ranks retrieval and seeds
                       few-shot examples for similar future queries
      │
      ▼
 injection defenses    ingestion-time flagging, data/instruction delimiting,
                       a heuristic+classifier gate on user input
      │
      ▼
 interfaces            CLI (ask / chat / feedback / status) and a browser UI
                       (finny serve) streaming the same pipeline live via SSE
```

Three roles: `ceo` (everything), `cto` (everything except headcount/compensation), `analyst` (financial statements only — no narrative text, no headcount). The headcount data is real, public Apple disclosure — I chose to classify it as restricted anyway so RBAC has something genuine to demonstrate without inventing confidential data (see `data/raw/README.md`).

## How the pieces fit — the one invariant that matters

Every retrieval path (`lookup_metric`, `search_filings`) takes the caller's `PermissionContext` and filters *inside* the SQL/Chroma query itself — `WHERE sensitivity IN (...)`, `where={"sensitivity": {"$in": [...]}}`. A restricted row or chunk is never fetched for a role that lacks the tag. That's the answer to "how do you know it won't leak": there's structurally nothing to leak, because the unauthorized data never entered the process, let alone the LLM's context.

The same principle extends to *combinations*: a role can individually hold `public_financial` and still be refused a computed metric like revenue-per-employee, because that computation also needs `headcount_compensation` — the guard checks the full requirement set before running anything, so the model never gets a silently-partial result to reason from.

Everything downstream — the agent loop, the feedback loop, the web UI — is a thin caller on top of that one enforcement point, not a second place RBAC has to be re-implemented. The LangGraph agent, for instance, never touches SQLite/Chroma directly; it can only see whatever a tool call returned, so RBAC holds regardless of what the model decides to do, including under a prompt-injection attempt (tools are closures over `PermissionContext` — there's no argument name the model could supply to change its own role).

## The feedback loop, concretely

- Every answer gets a `Query ID`. `finny feedback <id> --up/--down/--correct "..."` records a rating.
- **Re-ranking**: a downvote on a chunk/metric penalizes it for future *similar* queries (cosine similarity ≥ 0.45 on the query text, not exact match); an upvote boosts it. Verified live: downvoting the top-3 `search_filings` hits for a query and re-running it displaced all three.
- **Few-shot memory**: upvoted or corrected (query, answer) pairs are embedded and the most similar ones get injected as reference examples for new similar questions — role-scoped, since a `ceo`'s example could legitimately contain headcount figures a lower role must never see, even as a "reference."

## Prompt injection defenses (bonus)

Three layers sharing one pattern detector: chunks are flagged (never silently stripped) at ingestion time if they contain instruction-like text; retrieved content is wrapped in explicit `<doc_chunk>` data-delimiter tags with a system-prompt rule that it's never to be treated as instructions; and raw user questions are gated before the agent loop starts by a heuristic + a small Groq classifier. Worth mentioning on the call: an early version let the classifier veto *any* heuristic match, and live testing showed it would call a textbook injection attempt "SAFE" when padded with a plausible-sounding request. Fixed by making unambiguous heuristic matches classifier-proof — the classifier only gets a vote on genuinely ambiguous phrasing.

## What I'd do differently with more time

- Real auth (session tokens, not a `--role` flag) — the single biggest gap between this and a real multi-user product. Deliberately out of scope for a CLI/demo-scale assignment.
- Incremental re-indexing — right now every `build-index` run re-embeds and re-summarizes everything, with no diffing against what's already indexed.
- A second issuer's filings, to pressure-test whether the PDF section-header parsing generalizes or was just fit to Apple's specific 10-K layout.
- Finish documenting the web frontend (`finny serve`) to the same depth as everything above — it's built and working, just less thoroughly written up than the CLI path.

## Where it breaks at 100× the data or users

- **Single-node SQLite + Chroma.** Both are embedded, single-process stores. At 100× concurrent users, SQLite's writer lock serializes every feedback/audit-log write; Chroma's persistent client has no clustering story at all. Fix: a real Postgres for structured data and a hosted/clustered vector store — meaningful infrastructure, not a config flag.
- **Synchronous per-query Groq calls, no queueing.** The web server runs each request in a thread for the full duration of a possibly multi-step tool-calling loop. At 100× traffic, thread-pool exhaustion and Groq's per-key rate limits both bite before the data layer does. Fix: a real task queue or async client, plus caching for repeated questions.
- **PDF section-splitting is filing-format-specific.** The section-header detection is tuned to how Apple's actual 10-K/10-Q renders in `pdfplumber`. At 100× the *issuers* — not just more Apple filings — a differently-formatted filing wouldn't error, it would silently mis-chunk into one giant "General Overview" section, degrading retrieval without ever surfacing as a bug. This is the correctness risk that's easiest to miss when you've only tested against one issuer.
- **No real auth/identity.** Role is a request parameter, not an authenticated session. The RBAC *filtering* is real and correct; the *identity* feeding into it isn't, at 100× users sharing one deployment.
- **Full-rebuild ingestion, no diffing.** Every `build-index` run deletes and recreates the metrics table and the entire vector collection. Fine at 4 filings; at 100× the corpus, every run costs as much as the first one.
- **Local embedding throughput.** CPU-based `sentence-transformers` is the first real wall at scale — well before Groq limits or database contention. The fix (GPU inference or a hosted embedding endpoint) reopens the "no external embedding API" constraint this project deliberately chose to avoid, so it's a genuine trade-off, not a free win.
