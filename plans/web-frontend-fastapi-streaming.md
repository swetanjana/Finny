# Finny Web Frontend — FastAPI + Live Tool-Call Streaming

## Context

Phases 0–5 (ingestion, understanding layer, RBAC, LangGraph agent, feedback loop) are done and verified; Phase 7 has a partial eval gate and setup script. CLAUDE.md's own interface decision says a web wrapper is a "stretch add-on only after CLI+RBAC+feedback are solid — not before," which is now true. The goal here is a frontend whose primary job is to **make the backend's agility visible** — the multi-step LangGraph tool-calling loop, the RBAC filtering, and the Phase 5 feedback loop are all currently only observable through CLI text output or by reading code. A browser UI that streams each tool call live, lets you flip roles and watch what disappears, and lets you downvote/correct an answer and immediately re-ask to see retrieval change, turns those invisible backend properties into a 2-minute live demo.

Two decisions were confirmed with the user before this plan was drafted:
1. **Stack**: FastAPI + a single self-contained vanilla HTML/CSS/JS page (no Node, no build step, no CDN dependency — avoids the offline/sandboxed-network issues already seen with `yfinance`). Not Streamlit.
2. **Streaming is required**, not just a synchronous request/response — tool calls must appear in the UI as they happen, not all at once at the end.

This was researched and validated by a Plan subagent against the actual current code, then double-checked directly (re-read `agent/loop.py` and `interface/cli.py` in full to confirm exact signatures/behavior below). One refinement was made on top of the subagent's draft: tool-call-start/result correlation on the frontend should use the LangChain tool-call `id` (already available and already used the same way by `_extract_tool_results()`), not FIFO order — a single agent turn can contain more than one tool call in one LLM response, and `id`-based correlation is barely more code and strictly more correct.

## Guiding constraint

**The web layer is a thin wrapper only.** It must not reimplement or re-check RBAC, agent orchestration, or feedback logic — it calls the exact same functions `interface/cli.py` already calls (`PermissionContext`, `finny.rbac.tools`, `finny.feedback.store`, `finny.understanding.db`), plus one new streaming sibling of `run_agent()`. `run_agent()` itself must end up byte-behaviorally unchanged (verified by `evals/agent_cases.py`, which imports and calls it directly, and must keep passing unmodified).

---

## 1. Dependencies

```
uv add fastapi "uvicorn[standard]"
```
- No `jinja2` (no server-side templating — the HTML is one static file, all dynamic data comes from the JSON API).
- No `python-multipart` (no HTML forms, JS submits JSON via `fetch()`).
- No `sse-starlette` (FastAPI's own `StreamingResponse` is enough for hand-framed SSE).
- This is the project's first non-CLI-facing runtime dependency — worth one sentence in the Explainaibility.md write-up.

---

## 2. File layout

```
src/finny/interface/
├── cli.py                      # +1 new `serve` command; _extract_retrieved_ids moves out (see below)
├── status.py                   # NEW: shared status-summary reader (used by cli.status() and the web route)
└── web/
    ├── __init__.py
    ├── app.py                  # NEW: create_app() FastAPI factory — static mount + router
    ├── routes.py                # NEW: /api/ask (SSE), /api/feedback, /api/metrics, /api/status
    ├── schemas.py                # NEW: Pydantic request/response models
    └── static/
        └── index.html            # NEW: entire frontend, self-contained (inline <style>/<script>)
```

Small supporting refactors (behavior-preserving, not new logic):
- Move `_extract_retrieved_ids()` from `interface/cli.py` into `feedback/store.py` (it's feedback-domain logic — flattening retrieved ids for a feedback record — not CLI-domain logic). Both `cli.py` and `web/routes.py` import it from there. `cli.py`'s call site (`_render_answer`) just changes its import.
- Extract `interface/cli.py`'s `status()` body (reading `ingestion_summary.json`, building the summary dict) into `interface/status.py::get_status_summary(root_dir) -> Optional[dict]`. `cli.status()` becomes a thin `console.print`-only caller; `web/routes.py`'s `/api/status` calls the same function. Avoids duplicating the JSON path/parsing logic.

---

## 3. `agent/loop.py` changes — `_prepare_turn()` + `stream_agent()`

Refactor `run_agent()`'s current lines 106–137 (api-key resolution → graph cache lookup/build → few-shot injection → initial message construction) into:

```python
def _prepare_turn(
    context: PermissionContext, user_query: str, api_key: Optional[str]
) -> Union[Tuple[Any, List[Any]], Dict[str, Any]]:
    """Returns (graph, initial_messages) on success, or an error dict in run_agent()'s
    return shape ({"answer": ..., "tool_results": []}) if GROQ_API_KEY is missing.
    Shared by run_agent() and stream_agent() so there is exactly one place that
    decides how a turn starts — graph caching, few-shot injection, system prompt."""
```

`run_agent()` becomes: call `_prepare_turn()`, return early if it got the error-dict shape, else `graph.invoke(...)` and the existing post-processing — unchanged logic, just relocated. This is a pure refactor with no behavior change.

New sibling function, same file:

```python
def stream_agent(
    context: PermissionContext, user_query: str, api_key: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    """Same turn as run_agent(), but yields one event dict per LangGraph super-step
    via CompiledStateGraph.stream(stream_mode="updates") instead of blocking for the
    final answer. Used only by web/routes.py — the CLI keeps using run_agent()."""
```

Event dicts yielded (consumed by `web/routes.py`, not exposed to the CLI):
- `{"type": "tool_call_started", "call_id": str, "tool": str, "arguments": dict}` — one per `tool_calls` entry on an `"agent"` update's AIMessage.
- `{"type": "tool_call_result", "call_id": str, "tool": str, "result": dict}` — one per ToolMessage on the matching `"tools"` update. `call_id` = `msg.tool_call_id`, giving the frontend exact start↔result correlation (same mechanism `_extract_tool_results()` already uses via `call_by_id`).
- `{"type": "final_answer", "answer": str}` — when an `"agent"` update's message has content and no further tool calls.
- `{"type": "error", "message": str}` — on missing API key, `GraphRecursionError`, or any other exception; mirrors `run_agent()`'s three error paths exactly.
- `{"type": "done", "tool_results": [...]}` — last event; `tool_results` built by calling the existing `_extract_tool_results()` on the full accumulated message list (initial + every agent/tool message seen), so the sources table is constructed identically to how `run_agent()` builds it today.

`stream_agent()` does **not** touch SQLite (no `record_query` call) — same separation of concerns as `run_agent()`, which also doesn't log. Logging is the caller's job (today: `cli._render_answer()`; going forward: `web/routes.py`'s `/api/ask` handler, symmetric to that).

---

## 4. API endpoints (`web/routes.py`, `APIRouter()`, mounted at `/api`)

### `POST /api/ask` — streaming
- Request: `{"role": str, "query": str}` (`schemas.AskRequest`).
- Validates role via `PermissionContext(role)`, catching `UnknownRoleError` → `HTTPException(400, ...)`.
- **Plain `def` route (not `async def`)**: FastAPI runs sync path functions in a threadpool automatically, which is the correct way to avoid blocking the event loop with `stream_agent()`'s blocking Groq HTTP calls, with zero extra dependencies (no `anyio.to_thread` plumbing needed). Note this explicitly in the docs as the honest single-worker-demo answer to "how would this scale to concurrent users."
- Response: `StreamingResponse(media_type="text/event-stream")`, each event as `data: {json}\n\n`. Consumed via browser `fetch()` + `ReadableStream.getReader()` — **not** the native `EventSource` API, since `EventSource` is GET-only and can't carry the JSON body (role + query).
- After the generator's last non-`done` event, the handler itself (not `stream_agent()`) calls `feedback_store.record_query(conn, role, query, final_answer, _extract_retrieved_ids(tool_results))` and injects the resulting `query_id` into the final `done` frame before sending it. This is the one piece of "logging glue" the web layer legitimately owns — the same thing `cli._render_answer()` already does after `run_agent()` returns.

### `POST /api/feedback`
- Request: `{"query_id": int, "rating": Optional["up"|"down"], "correction_text": Optional[str]}`.
- Mirror `cli.py`'s `feedback()` validation exactly (confirmed by re-reading `cli.py` lines 143–174): reject if neither `rating` nor `correction_text` given; if `rating` is omitted but `correction_text` is present, default `rating = "down"` (matches the CLI's `rating = "up" if up else "down"` when only `--correct` is passed).
- Delegates straight to `feedback_store.record_feedback(conn, CHROMA_DIR, query_id, rating, correction_text=...)`. `FeedbackTargetNotFound` → 404.

### `GET /api/metrics?role=...`
- Thin wrapper over `rbac.tools.list_available_metrics(PermissionContext(role))` — calls the exact RBAC-filtered tool the agent itself uses, but directly (no Groq call needed), so the UI sidebar can show "what this role can see" instantly on role switch. 400 on unknown role.

### `GET /api/status`
- Wraps the new `interface/status.py::get_status_summary()` (see §2).

---

## 5. Frontend (`web/static/index.html`, single file)

- **Sidebar**: role `<select>` (ceo/cto/analyst, default `analyst` — starts on the most-restricted view so the RBAC demo is visible immediately), a live "metrics visible to this role" list (refetched from `/api/metrics` on role change — flipping the dropdown visibly changes this list with zero LLM cost), a small dataset-status block from `/api/status`.
- **Main panel**: chat-style question input + scrollable answer history (same idea as `finny chat`, in a browser).
- **Live tool-call trace**: as soon as the SSE stream starts, each `tool_call_started` event appends a pending row (`🔧 calling {tool}({arguments})…`); the matching `tool_call_result` event (correlated by `call_id`) updates that same row in place (`✅ N results` / `🚫 restricted: {reason}` / `⚠️ error`, colored by `status`). This is the literal "showcase agility" feature.
- **Final answer card**: renders `final_answer`, then — once `done` arrives — a sources table (Tool / Args / Status / Retrieved), same shape as the CLI's rich table, built from `done.tool_results`.
- **Feedback controls**: once `done` carries `query_id`, show it plus inline 👍 / 👎 / "✏️ correct" (text input + submit), wired to `POST /api/feedback`. This makes the Phase 5 before/after demo (ask → downvote/correct → re-ask → watch the trace panel change) a first-class, one-page flow instead of a manual CLI multi-step process.
- **JS**: no framework — plain functions (`askQuestion()` does the `fetch()` + stream-reading + event dispatch, `submitFeedback()`, `refreshMetrics(role)`, `refreshStatus()`). All inline, no external `<script src>`/`<link>`/`@import` — verified offline-safe.

---

## 6. New CLI command

```python
@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on source changes (dev only)"),
):
    """Run the FastAPI web UI — a long-lived process, unlike each CLI invocation."""
    import uvicorn
    uvicorn.run("finny.interface.web.app:create_app", factory=True, host=host, port=port, reload=reload)
```
No change to `pyproject.toml`'s `[project.scripts]` — `finny serve` reuses the existing single entrypoint.

---

## 7. Documentation updates

- **Explainaibility.md**: new section (§33 onward, following §32) covering: architecture diagram (browser ↔ SSE ↔ `stream_agent()` ↔ existing RBAC/agent/feedback modules), why `_prepare_turn()` avoids duplicating `run_agent()` (cite that `evals/agent_cases.py` keeps passing unmodified as proof), the SSE event schema, the endpoint list, and — as a concrete, *measured* callout — the caching payoff: a `finny serve` process reuses `_graph_cache` and the `@lru_cache`d embedding function/Chroma client (Phase 5) *across* HTTP requests, unlike each `finny ask` CLI invocation which is a fresh process paying the ~115ms graph-compile + model-load cost every time. Include actual observed timings from verification step 5 below, not an estimate. Also record the feedback-loop live-demo walkthrough and known limitations (single sync worker, no auth — same acknowledged gap as the CLI's `--role` flag, no SSE reconnect).
- **CLAUDE.md**: update "Current state" with the new `web/` module, `finny serve` command, and confirm the "Interface... stretch add-on" line is now built; add a brief "Phase 8 — Web Frontend" entry to the Implementation Phases list in the same one-paragraph style as Phases 0–7.

---

## 8. Verification plan

1. `uv sync`; `uv run finny --help` lists `serve` alongside all existing commands (import didn't break).
2. `uv run finny serve --port 8000` in background; `curl -s :8000/` returns the HTML; `curl -s :8000/api/status` returns the ingestion summary.
3. **RBAC, no LLM needed**: `curl ':8000/api/metrics?role=analyst'` vs `cto` vs `ceo` — `Headcount` present only for `ceo` (same assertion as `evals/cases.py`'s `discovery-headcount-hidden-from-cto-analyst` case, now checked through the HTTP layer too).
4. **Streaming is actually incremental**: `curl -N -X POST :8000/api/ask -d '{"role":"cto","query":"What was net income in FY2024?"}'` piped through a timestamping filter — confirm `tool_call_started`/`tool_call_result`/`final_answer`/`done` frames arrive at visibly different times, not as one blob. This is the core claim and must be shown, not assumed.
5. **Caching payoff, measured**: two consecutive `/api/ask` calls (different queries, same role) against one running server — compare time-to-first-SSE-frame between request 1 and request 2; record actual numbers into Explainaibility.md.
6. **Feedback loop live, via curl**: ask a `search_filings` question, capture `query_id` from `done`; `POST /api/feedback {"query_id": N, "rating": "down"}`; re-ask a related query; diff `tool_call_result` ids/order against the first run.
7. **Browser smoke test**: role switch updates the sidebar; a question visibly fills in the trace panel step by step before the final answer; thumbs-down/correct via the UI; re-ask shows a different trace. Manual pass (this is the one step that needs an eyeball, not curl).
8. **Regression check (most important)**: `uv run finny ask --role ceo "..."` still works identically; `uv run python -m evals.run_evals` (17 cases) and `uv run python -m evals.run_evals --agent` (21 total, quota permitting) both still pass — proves the `_prepare_turn()` extraction didn't change `run_agent()`'s behavior.
9. **Offline-safety check**: confirm `index.html` has no external `<script src>`/`<link>`/`@import` — loads and works with network access restricted to the local Finny server itself.
