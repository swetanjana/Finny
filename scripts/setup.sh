#!/usr/bin/env bash
# Bootstraps Finny after a fresh `git clone`: installs dependencies, sets up
# .env, and regenerates everything derived from data/raw/ (ingestion JSONL,
# SQLite metrics/audit/feedback tables, Chroma narrative index, per-filing
# understanding summaries). Safe to re-run any time — build-index is idempotent.
#
# Usage: ./scripts/setup.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "==> Checking for uv..."
if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' is not installed. Install it first:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    echo "  (see https://docs.astral.sh/uv/getting-started/installation/)" >&2
    exit 1
fi

echo "==> Installing dependencies (uv sync)..."
uv sync

if [ ! -f .env ]; then
    echo "==> No .env found — creating one from .env.example."
    cp .env.example .env
    echo "    Edit .env and set GROQ_API_KEY before running 'finny ask' / 'finny chat'."
else
    echo "==> .env already exists, leaving it untouched."
fi

if ! grep -qE '^GROQ_API_KEY=.+[^[:space:]]' .env || grep -q '^GROQ_API_KEY=your_groq_api_key_here' .env; then
    echo "    WARNING: GROQ_API_KEY in .env looks unset — 'finny ask'/'finny chat' will fail until it's filled in."
    echo "    (data ingestion and indexing below don't need it and will still work.)"
fi

if [ ! -d data/raw ] || [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
    echo "error: data/raw/ is missing or empty — this repo expects the raw filings to be committed alongside the code. Aborting." >&2
    exit 1
fi

echo "==> Building the index (ingestion + SQLite + Chroma + summaries)..."
uv run finny build-index

echo "==> Verifying with 'finny status'..."
uv run finny status

echo
echo "==> Running the deterministic eval gate (RBAC + metric accuracy)..."
uv run python -m evals.run_evals

echo
echo "Setup complete. Try:"
echo "  uv run finny ask --role cto \"What was net income in FY2024?\""
echo "  uv run finny chat --role analyst"
echo "  uv run python -m evals.run_evals --agent   # + live end-to-end NL suite (needs GROQ_API_KEY)"
