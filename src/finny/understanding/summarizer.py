"""Per-filing understanding files: one markdown per source document, precomputed at
index time via a single Groq call each, so the agent's runtime context stays cheap.

Numeric figures are never LLM-generated — the "Key Metrics" / "Key Sections" block is
built deterministically from the extracted records. Groq is only asked to write the
narrative overview paragraph, which keeps the summary auditable against source data.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from groq import Groq

SUMMARY_MODEL = "llama-3.3-70b-versatile"

HEADLINE_METRIC_KEYWORDS = [
    "net sales", "total net sales", "net income", "total assets", "total liabilities",
    "cash and cash equivalents", "earnings per share", "gross margin", "operating income",
    "research and development", "stockholders equity", "shareholders equity",
    "total revenue", "operating expenses", "diluted", "basic",
]

MAX_METRIC_ROWS = 40
MAX_NARRATIVE_CHARS = 6000


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _group_by_file(records: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(rec.get("file", "unknown"), []).append(rec)
    return grouped


def _select_headline_metrics(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    selected = []
    for rec in records:
        key = (rec["metric"], rec["period"])
        if key in seen:
            continue
        name_lower = rec["metric"].lower()
        if any(kw in name_lower for kw in HEADLINE_METRIC_KEYWORDS):
            seen.add(key)
            selected.append(rec)
        if len(selected) >= MAX_METRIC_ROWS:
            break
    return selected


def _select_narrative_excerpts(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One representative chunk per unique section, in first-seen order, capped by length."""
    seen_sections = set()
    excerpts = []
    total_chars = 0
    for rec in records:
        section = rec.get("section", "General Overview")
        if section in seen_sections:
            continue
        seen_sections.add(section)
        content = rec["content"][:600]
        if total_chars + len(content) > MAX_NARRATIVE_CHARS:
            break
        total_chars += len(content)
        excerpts.append({**rec, "content": content})
    return excerpts


def _call_groq_overview(client: Groq, file_name: str, doc_type: str, fiscal_year: str, context_text: str) -> str:
    prompt = (
        f"You are generating a concise internal understanding-file overview for the financial "
        f"document '{file_name}' (doc_type={doc_type}, fiscal_year={fiscal_year}). "
        f"Based ONLY on the excerpts below, write a 3-5 sentence plain-prose overview of what "
        f"this document covers. Do not invent numbers not present in the excerpts, and do not "
        f"repeat the excerpts verbatim.\n\nExcerpts:\n{context_text}"
    )
    try:
        response = client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": "You write terse, accurate financial-document overviews."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"_(Groq summary unavailable: {exc})_"


def generate_summaries(
    processed_dir: Path,
    summaries_dir: Path,
    manifest_by_file: Dict[str, Dict[str, Any]],
    api_key: str | None,
) -> List[str]:
    summaries_dir.mkdir(parents=True, exist_ok=True)
    client = Groq(api_key=api_key) if api_key else None

    metrics_by_file = _group_by_file(_iter_jsonl(processed_dir / "financial_metrics.jsonl"))
    narratives_by_file = _group_by_file(_iter_jsonl(processed_dir / "narrative_chunks.jsonl"))

    written: List[str] = []
    all_files = set(metrics_by_file) | set(narratives_by_file)

    for file_name in sorted(all_files):
        manifest_info = manifest_by_file.get(file_name, {})
        doc_type = manifest_info.get("doc_type", "unknown")
        fiscal_year = str(manifest_info.get("fiscal_year", ""))

        metric_records = metrics_by_file.get(file_name, [])
        narrative_records = narratives_by_file.get(file_name, [])
        sensitivities = sorted({r["sensitivity"] for r in metric_records + narrative_records})

        lines = [
            f"# Understanding File: {file_name}",
            "",
            f"- **doc_type**: {doc_type}",
            f"- **fiscal_year**: {fiscal_year}",
            f"- **sensitivity tags present**: {', '.join(sensitivities)}",
            f"- **record count**: {len(metric_records) + len(narrative_records)} "
            f"({len(metric_records)} metrics, {len(narrative_records)} narrative chunks)",
            "",
            "## Overview",
            "",
        ]

        if narrative_records:
            excerpts = _select_narrative_excerpts(narrative_records)
            context_text = "\n\n".join(f"[{e['section']}] {e['content']}" for e in excerpts)
            overview = _call_groq_overview(client, file_name, doc_type, fiscal_year, context_text) if client else (
                "_(No GROQ_API_KEY configured; overview skipped.)_"
            )
            lines.append(overview)
            lines.append("")
            lines.append("## Key Sections")
            lines.append("")
            for e in excerpts:
                lines.append(f"- **{e['section']}** (page {e.get('source_page', '?')}): {e['content'][:160]}...")
        else:
            headline = _select_headline_metrics(metric_records)
            context_text = "\n".join(f"{m['metric']} ({m['period']}): {m['value']} {m.get('unit', '')}" for m in headline)
            overview = _call_groq_overview(client, file_name, doc_type, fiscal_year, context_text) if client else (
                "_(No GROQ_API_KEY configured; overview skipped.)_"
            )
            lines.append(overview)
            lines.append("")
            lines.append("## Key Metrics")
            lines.append("")
            lines.append("| Metric | Period | Value | Unit | Sensitivity |")
            lines.append("|---|---|---|---|---|")
            for m in headline:
                lines.append(f"| {m['metric']} | {m['period']} | {m['value']:,.0f} | {m.get('unit', '')} | {m['sensitivity']} |")

        # Phase 6: surface every instruction-like-pattern flag raised at ingestion
        # time — auditable here rather than silently dropped, even though the
        # flagged text itself was never altered or excluded from the index.
        flagged_chunks = [r for r in narrative_records if r.get("flagged")]
        if flagged_chunks:
            lines.append("")
            lines.append("## Security — Flagged Content")
            lines.append("")
            lines.append(
                f"{len(flagged_chunks)} of {len(narrative_records)} narrative chunk(s) in this "
                "document matched an instruction-like pattern during ingestion. Content was kept "
                "unmodified in the index; flag it for review before treating as reliable narrative."
            )
            lines.append("")
            for r in flagged_chunks:
                reasons = ", ".join(r.get("flag_reasons", []))
                lines.append(f"- **{r.get('section', '?')}** (page {r.get('source_page', '?')}), matched: `{reasons}`")
                lines.append(f"  > {r['content'][:200]}...")

        lines.append("")
        lines.append("## Provenance")
        lines.append("")
        lines.append(f"- source file: `{file_name}`")
        lines.append(f"- manifest doc_type: `{doc_type}`, fiscal_year: `{fiscal_year}`")

        # Use the full filename (extension included) as the stem: source files like
        # 10K-2023.pdf and 10K-2023.xls share a Path.stem and would otherwise collide.
        out_path = summaries_dir / f"{file_name.replace('.', '_')}.md"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        written.append(str(out_path))

    return written
