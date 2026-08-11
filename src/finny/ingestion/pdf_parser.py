"""PDF parser for SEC 10-K and 10-Q filings with section-aware chunking."""

from pathlib import Path
from typing import Any, Dict, List
import re
import hashlib
import pdfplumber

from finny.security.injection_patterns import detect_injection_patterns


class PDFIngestor:
    """Extracts text from 10-K and 10-Q PDFs using section-aware chunking."""

    SECTION_PATTERNS = [
        re.compile(r"^\s*(ITEM\s+1A?B?\.\s+[^.\n]+)", re.IGNORECASE),
        re.compile(r"^\s*(ITEM\s+[0-9]{1,2}[A-Z]?\.\s+[^.\n]+)", re.IGNORECASE),
        re.compile(r"^\s*(PART\s+[I|V|X]+\b[^.\n]*)", re.IGNORECASE),
        re.compile(r"^\s*(MANAGEMENT'S\s+DISCUSSION\s+AND\s+ANALYSIS[^.\n]*)", re.IGNORECASE),
        re.compile(r"^\s*(RISK\s+FACTORS\b[^.\n]*)", re.IGNORECASE),
        re.compile(r"^\s*(BUSINESS\b[^.\n]*)", re.IGNORECASE),
        re.compile(r"^\s*(FINANCIAL\s+STATEMENTS[^.\n]*)", re.IGNORECASE),
    ]

    def __init__(self, target_chunk_size: int = 1200, overlap: int = 150):
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap

    def process_file(self, file_path: Path, manifest_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extracts text per page/section and splits into section-aware narrative chunks."""
        chunks: List[Dict[str, Any]] = []
        doc_type = manifest_info.get("doc_type", "10-K")
        fiscal_year = str(manifest_info.get("fiscal_year", ""))
        default_sensitivity = manifest_info.get("default_sensitivity", "public_narrative")

        current_section = "General Overview"

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text or not text.strip():
                    continue

                lines = text.split("\n")
                page_text_blocks = []

                for line in lines:
                    stripped_line = line.strip()
                    if not stripped_line:
                        continue

                    # Detect section heading
                    detected_heading = self._detect_section_heading(stripped_line)
                    if detected_heading:
                        current_section = detected_heading

                    page_text_blocks.append(stripped_line)

                page_text = " ".join(page_text_blocks)
                if not page_text:
                    continue

                # Split page text into chunks of target size
                page_chunks = self._chunk_text(page_text)

                for idx, chunk_text in enumerate(page_chunks):
                    chunk_id_hash = hashlib.sha256(
                        f"{file_path.name}_p{page_num}_c{idx}_{chunk_text[:30]}".encode("utf-8")
                    ).hexdigest()[:16]

                    # Phase 6: flag instruction-like patterns for audit — never
                    # strip them, since a filing's actual text should never be
                    # silently altered; downstream (prompt + understanding file)
                    # decides how to handle a flagged chunk.
                    flag_reasons = detect_injection_patterns(chunk_text)

                    chunk_rec = {
                        "id": f"chunk_{chunk_id_hash}",
                        "type": "narrative_chunk",
                        "doc_type": doc_type,
                        "fiscal_year": fiscal_year,
                        "section": current_section,
                        "source_page": page_num,
                        "content": chunk_text,
                        "sensitivity": default_sensitivity,
                        "file": file_path.name,
                        "flagged": bool(flag_reasons),
                        "flag_reasons": flag_reasons,
                    }
                    chunks.append(chunk_rec)

        return chunks

    def _detect_section_heading(self, line: str) -> str | None:
        """Checks if a line matches known SEC filing section header patterns."""
        for pattern in self.SECTION_PATTERNS:
            match = pattern.match(line)
            if match:
                heading = match.group(1).strip()
                # Ensure heading is reasonably concise (not an entire paragraph)
                if len(heading) < 120:
                    return heading
        return None

    def _chunk_text(self, text: str) -> List[str]:
        """Splits page text into overlapping windows of roughly target_chunk_size characters."""
        if len(text) <= self.target_chunk_size:
            return [text]

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.target_chunk_size
            if end >= text_length:
                chunks.append(text[start:].strip())
                break

            # Try to break at a sentence boundary or space near end
            break_pos = text.rfind(". ", start + self.target_chunk_size // 2, end)
            if break_pos == -1:
                break_pos = text.rfind(" ", start + self.target_chunk_size // 2, end)
            if break_pos == -1:
                break_pos = end

            chunk = text[start:break_pos + 1].strip()
            if chunk:
                chunks.append(chunk)

            start = max(start + 1, break_pos + 1 - self.overlap)

        return chunks
