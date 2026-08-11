# Finny — Financial-Data Agent

A role-based-access-controlled agent over Apple Inc.'s public SEC filings (10-K/10-Q PDFs + Excel financial statements). Ask natural-language questions, get answers grounded in retrieved data with citations, and the system learns from thumbs-up/down and corrections.

Built for the Azentio AI Agent Developer tech-round assignment — full brief in [AI_Agent_Dev_Assignment.md](AI_Agent_Dev_Assignment.md). For the walkthrough — what was built, how the pieces fit, what I'd do differently, what breaks at scale — see [WALKTHROUGH.md](WALKTHROUGH.md). (`Explainaibility.md` is a much longer personal build log kept during development, not written for external readers.)

## Quick start

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and a free [Groq API key](https://console.groq.com/keys).

```bash
./scripts/setup.sh
```

This installs dependencies, creates `.env` from `.env.example` (edit it and set `GROQ_API_KEY` before asking questions — ingestion/indexing don't need it), rebuilds the index from `data/raw/`, and runs the deterministic eval gate. Or do it by hand:

```bash
uv sync
cp .env.example .env        # then edit .env and set GROQ_API_KEY
uv run finny build-index    # ingest data/raw/ -> SQLite metrics + Chroma narrative index + per-filing summaries
uv run finny status         # sanity check
```

`data/raw/` (the 10-K/10-Q PDFs, their Excel exports, a curated metrics workbook, and `manifest.csv`) is committed to the repo, so `build-index` is the only thing you need to run — there's nothing to download. It's idempotent: safe to re-run any time, always reflects the current contents of `data/raw/`.

## Using it

```bash
uv run finny ask --role cto "What was net income in FY2024?"
uv run finny chat --role analyst          # multi-turn REPL; follow-ups use prior turns as context
uv run finny feedback 12 --down --correct "..."   # correct a past answer (query ID printed after each answer)
uv run finny serve --port 8000            # browser UI: FastAPI + live SSE tool-call streaming
```

Roles: `ceo`, `cto`, `analyst` — see [Roles & RBAC](#roles--rbac) below.

## Config / keys

Only one secret: `GROQ_API_KEY` in `.env` (copy from `.env.example`), used for the agent's reasoning model, the per-filing summary generation, and the Phase 6 injection classifier. Everything else — embeddings (`sentence-transformers`, local), storage (SQLite + Chroma, local files under `data/`) — runs offline with no other keys or external services.

## Roles & RBAC

| Role | Sees |
|---|---|
| `ceo` | everything — filing narrative, financial statements, headcount/compensation |
| `cto` | filing narrative + financial statements — **not** headcount/compensation |
| `analyst` | financial statements only — no filing narrative/MD&A/risk text, no headcount |

Enforcement is at the data-retrieval layer (`src/finny/rbac/tools.py`), not the prompt: every SQL/Chroma query filters by the caller's allowed sensitivity tags before anything reaches the LLM, so there's nothing to leak even for a restricted-name lookup (a role lacking `headcount_compensation` is refused, not silently given zero rows) or a computed metric that would require combining a permitted and a restricted source (e.g. revenue-per-employee is refused outright for `cto`/`analyst`, never computed from partial data). See `Explainaibility.md` §13–17 for the design and verification.

**Note on the data**: `apple_metrics_curated.xlsx`'s `restricted_headcount` sheet is public-source data (Apple discloses headcount) that this project *intentionally* classifies as restricted (`headcount_compensation`), so RBAC has something real to demonstrate without inventing confidential data. See `data/raw/README.md`.

## What's implemented

- **Ingestion** (`src/finny/ingestion/`): PDF → section-aware chunks (`Item 1A. Risk Factors`, MD&A, etc.), Excel → row-level metric records, both tagged with sensitivity and full provenance.
- **Understanding layer** (`src/finny/understanding/`): SQLite for exact metric lookups, Chroma (local `sentence-transformers` embeddings) for narrative semantic search, one Groq-generated markdown summary per filing.
- **RBAC** (`src/finny/rbac/`): role → allowed-tags, enforced at the query layer, every call audit-logged.
- **Agent** (`src/finny/agent/`): explicit LangGraph tool-calling loop (`lookup_metric`, `search_filings`, `list_available_metrics`, `get_live_stock_price`) over a Groq model, multi-turn chat memory, cached graph compilation.
- **Feedback loop** (`src/finny/feedback/`): thumbs-up/down/correction re-ranks future retrieval for similar queries and injects upvoted/corrected answers as few-shot examples.
- **Prompt injection defenses** (`src/finny/security/`, `agent/injection_guard.py`): ingestion-time flagging of instruction-like text (never stripped, surfaced for audit), retrieved content wrapped in explicit data-not-instructions tags, and a heuristic+classifier gate on user input before it reaches the agent loop.
- **Web UI** (`src/finny/interface/web/`, `finny serve`): browser front end streaming each tool call live via SSE, so RBAC filtering and the feedback loop are visible in real time, not just in CLI text.

## Testing

```bash
uv run python -m evals.run_evals            # 23 deterministic cases: RBAC leak-proofing, metric accuracy, disambiguation, injection detection — no GROQ_API_KEY needed
uv run python -m evals.run_evals --agent     # + 4 live end-to-end NL cases through the real agent (needs GROQ_API_KEY)
uv run pytest evals/                         # same gate, pytest-native
```

## What breaks at 100×

(Full reasoning per point in [WALKTHROUGH.md](WALKTHROUGH.md).)

- **Single-node SQLite + Chroma**: fine for one demo user; concurrent writers (feedback, audit log) will contend, and there's no sharding/replication story.
- **Synchronous per-query Groq calls**: no request queueing, batching, or response caching — latency scales linearly with concurrent users, and there's no backpressure.
- **PDF section-splitting is filing-format-specific**: the section-header regexes are tuned to Apple's actual 10-K/10-Q layout; a different issuer's filing format would silently mis-chunk rather than fail loudly.
- **No real auth/identity**: role is a CLI flag / request parameter, not an authenticated session — fine for a demo, not for multiple real users sharing one deployment.
- **Full-rebuild ingestion**: `build-index` re-embeds and re-summarizes everything every run; there's no diffing, so a large corpus makes every ingest run as expensive as the first one.
- **Local embedding throughput**: `sentence-transformers` on CPU is the bottleneck once the narrative corpus is much larger than four filings — no batching tuning, no GPU path.
