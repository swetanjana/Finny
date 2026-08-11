"""Shared instruction-like pattern detector (Phase 6).

One regex list, used two places: ingestion-time flagging of narrative chunks
(`ingestion/pdf_parser.py`) and the pre-agent-loop heuristic gate on raw user
questions (`agent/injection_guard.py`). Kept here rather than duplicated so the
two call sites can't silently drift out of sync.

Deliberately regex-based and cheap (no network, no LLM) — it's a fast first
pass, not the final word. Ingestion never strips flagged content, it only
tags it for audit; the user-input gate escalates a heuristic match to a Groq
classifier before refusing (see `agent/injection_guard.py`), so an
over-broad pattern here costs a confirmation call, not a false refusal.
"""

import re
from typing import List, Tuple

_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("ignore_instructions", re.compile(r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE)),
    ("disregard_instructions", re.compile(r"disregard\s+(all\s+)?(the\s+)?(previous|prior|above|your)\s+(instructions|rules|prompt)", re.IGNORECASE)),
    ("role_override", re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE)),
    ("system_prefix", re.compile(r"^\s*system\s*:", re.IGNORECASE | re.MULTILINE)),
    ("new_instructions", re.compile(r"\bnew\s+instructions\s*:", re.IGNORECASE)),
    ("prompt_reveal", re.compile(r"\b(reveal|print|show|repeat)\s+(your|the)\s+(system\s+)?(prompt|instructions)\b", re.IGNORECASE)),
    ("jailbreak_persona", re.compile(r"\b(dan|do anything now)\b", re.IGNORECASE)),
    ("developer_mode", re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE)),
]

# Patterns whose wording is essentially unambiguous — no plausible legitimate
# financial question reads this way — so a match is authoritative on its own.
# Kept out of this set: patterns with a plausible innocent reading (e.g.
# "you are now" also completes "...looking at Apple's FY2024 numbers"), which
# the classifier gets to weigh in on rather than block outright. See
# `agent/injection_guard.py`: a HIGH_CONFIDENCE match is never overridden by
# the classifier — only escalated for a second opinion, matches outside this
# set are.
HIGH_CONFIDENCE_PATTERNS = frozenset(
    {"ignore_instructions", "disregard_instructions", "system_prefix", "new_instructions", "prompt_reveal", "jailbreak_persona"}
)


def detect_injection_patterns(text: str) -> List[str]:
    """Returns the names of every instruction-like pattern found in `text` (empty
    list if none). Pure/stateless — safe to call on ingestion chunks or live
    user queries alike."""
    return [name for name, pattern in _PATTERNS if pattern.search(text)]
