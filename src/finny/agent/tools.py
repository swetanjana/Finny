"""LangChain tool definitions for the Finny agent (LangGraph tool-calling loop).

Three tools wrap the RBAC-enforced data-layer functions from `finny.rbac.tools`
(the agent never talks to SQLite/Chroma directly, only through those RBAC-checked
functions). `get_live_stock_price` is the one unrestricted/public tool — it needs
no PermissionContext since Yahoo Finance market data isn't in the sensitivity
taxonomy at all.

`build_tools(context)` is a factory rather than module-level tool objects because
each tool must be bound to the caller's PermissionContext via closure — the model
supplies query arguments (metric name, search text, ...), never the role/permissions,
so there is no way for a prompt-injected tool call to widen its own access.
"""

import json
from typing import Any, Dict, List, Literal, Optional

import yfinance as yf
from langchain_core.tools import BaseTool, tool

from finny.rbac import tools as rbac_tools
from finny.rbac.context import PermissionContext


def get_live_stock_price(ticker: str = "AAPL") -> Dict[str, Any]:
    try:
        t = yf.Ticker(ticker)
        fast_info = t.fast_info
        price = fast_info.get("last_price") if hasattr(fast_info, "get") else getattr(fast_info, "last_price", None)
        currency = fast_info.get("currency") if hasattr(fast_info, "get") else getattr(fast_info, "currency", None)

        if price is None:
            hist = t.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])

        if price is None:
            return {"status": "error", "reason": f"Could not fetch a live price for '{ticker}'."}

        return {
            "status": "ok",
            "ticker": ticker.upper(),
            "price": round(float(price), 2),
            "currency": currency or "USD",
            "source": "Yahoo Finance (yfinance) — public market data, unrestricted by role",
        }
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}


def build_tools(context: PermissionContext) -> List[BaseTool]:
    """Returns the LangChain tool set for one query/session, bound to `context`."""

    @tool
    def lookup_metric(
        metric: str,
        period: Optional[str] = None,
        derived_metric_key: Optional[Literal["revenue_per_employee", "compensation_ratio"]] = None,
    ) -> str:
        """Exact lookup of a financial metric by name and optional fiscal period from
        Apple's structured filings data. Use for questions asking for a specific
        number, e.g. 'What was net income in FY2024?'.

        Apple's filings label total revenue as "Total net sales", not "revenue" —
        pass the exact line-item name where you know it (e.g. "Total net sales",
        not "revenue") to avoid matching an unrelated line item like "Services
        Revenue" (one segment) or "Deferred revenue" (a balance-sheet liability).

        If the result has status "ambiguous", your `metric` string matched more
        than one distinct line item and none was returned — call this again with
        one of the exact names in `candidates`, or call `list_available_metrics`.

        `derived_metric_key`: set ONLY if the user is asking for a computed metric
        that combines data across sensitivity tags (e.g. revenue per employee needs
        both financial and headcount data). Checks the caller's access to every
        required tag before anything is returned.
        """
        result = rbac_tools.lookup_metric(context, metric=metric, period=period, derived_metric_key=derived_metric_key)
        return json.dumps(result, default=str)

    @tool
    def search_filings(
        query_text: str,
        n_results: int = 4,
        doc_type: Optional[str] = None,
        fiscal_year: Optional[str] = None,
    ) -> str:
        """Semantic search over narrative text from Apple's 10-K/10-Q filings (risk
        factors, MD&A, business description, etc). Use for open-ended or qualitative
        questions, not for exact numbers."""
        result = rbac_tools.search_filings(
            context, query_text=query_text, n_results=n_results, doc_type=doc_type, fiscal_year=fiscal_year
        )
        # Phase 6: wrap retrieved narrative text in an explicit data delimiter so
        # it reads unambiguously as quoted filing content, never as instructions
        # — regardless of what the underlying PDF text happens to contain.
        for item in result.get("results", []):
            if "content" in item:
                item["content"] = (
                    f'<doc_chunk flagged="{item.get("flagged", False)}">\n{item["content"]}\n</doc_chunk>'
                )
        return json.dumps(result, default=str)

    @tool
    def list_available_metrics() -> str:
        """Lists the distinct metric names the caller's role is permitted to see.
        Use this if unsure what exact metric name to pass to lookup_metric."""
        result = rbac_tools.list_available_metrics(context)
        return json.dumps(result, default=str)

    @tool("get_live_stock_price")
    def get_live_stock_price_tool(ticker: str = "AAPL") -> str:
        """Fetches Apple's (AAPL) current market stock price from Yahoo Finance.
        Public market data — unrestricted by role."""
        return json.dumps(get_live_stock_price(ticker), default=str)

    return [lookup_metric, search_filings, list_available_metrics, get_live_stock_price_tool]
