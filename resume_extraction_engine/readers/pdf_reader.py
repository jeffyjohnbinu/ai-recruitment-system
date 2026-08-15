"""
PDF Reader
----------
Extracts raw text from PDF resumes, with layout-aware handling for:
  - single/multi-column text
  - tables (skills matrices, project tables, etc.)
  - embedded images (flagged, optionally OCR'd if pytesseract is available)

Primary engine: pdfplumber (layout aware).
Fallback engine: PyPDF2 (used if pdfplumber fails to open/parse a file).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger("resume_engine.pdf_reader")

try:
    import pdfplumber

    _HAS_PDFPLUMBER = True
except ImportError:  # pragma: no cover
    _HAS_PDFPLUMBER = False

try:
    import PyPDF2

    _HAS_PYPDF2 = True
except ImportError:  # pragma: no cover
    _HAS_PYPDF2 = False

try:
    import pytesseract

    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False


@dataclass
class PageResult:
    page_number: int
    text: str
    tables: List[List[List[str]]] = field(default_factory=list)
    had_images: bool = False
    column_count_guess: int = 1


@dataclass
class PDFReadResult:
    source_path: str
    pages: List[PageResult]
    engine_used: str
    warnings: List[str] = field(default_factory=list)

    @property
    def raw_text(self) -> str:
        """Concatenate all page text, tables rendered as pipe-separated rows."""
        chunks = []
        for page in self.pages:
            if page.text.strip():
                chunks.append(page.text)
            for table in page.tables:
                for row in table:
                    cells = [c if c else "" for c in row]
                    chunks.append(" | ".join(cells))
        return "\n".join(chunks)


class PDFReader:
    """Reads PDF resumes and returns structured, page-level text + tables."""

    def __init__(self, ocr_fallback: bool = True):
        self.ocr_fallback = ocr_fallback and _HAS_OCR

    def read(self, path: str | Path) -> PDFReadResult:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        if _HAS_PDFPLUMBER:
            try:
                return self._read_with_pdfplumber(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("pdfplumber failed for %s: %s. Falling back to PyPDF2.", path, exc)

        if _HAS_PYPDF2:
            return self._read_with_pypdf2(path)

        raise RuntimeError("No usable PDF backend available. Install pdfplumber or PyPDF2.")

    # ------------------------------------------------------------------ #
    # pdfplumber path (preferred — layout & table aware)
    # ------------------------------------------------------------------ #
    def _read_with_pdfplumber(self, path: Path) -> PDFReadResult:
        pages: List[PageResult] = []
        warnings: List[str] = []

        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = self._extract_multicolumn_text(page)
                tables = page.extract_tables() or []
                images = page.images or []

                if not text.strip() and not tables:
                    warnings.append(f"Page {i}: no extractable text or tables found.")

                if images:
                    if self.ocr_fallback and not text.strip():
                        ocr_text = self._ocr_page(page)
                        if ocr_text:
                            text = ocr_text
                            warnings.append(f"Page {i}: text recovered via OCR fallback.")
                    else:
                        warnings.append(
                            f"Page {i}: contains {len(images)} image(s); "
                            "image content not extracted as text."
                        )

                pages.append(
                    PageResult(
                        page_number=i,
                        text=text,
                        tables=tables,
                        had_images=bool(images),
                        column_count_guess=self._guess_column_count(page),
                    )
                )

        return PDFReadResult(
            source_path=str(path), pages=pages, engine_used="pdfplumber", warnings=warnings
        )

    def _extract_multicolumn_text(self, page) -> str:
        """
        Detect a 2-column resume layout and read left-column-then-right-column
        instead of pdfplumber's default top-to-bottom, left-to-right word
        order (which interleaves columns and scrambles sentences).

        Full-width lines (e.g. a centered name/contact header that spans both
        "columns") are kept intact in their natural top-to-bottom position
        instead of being sliced in half by a blind x-midpoint crop.
        """
        words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
        if not words:
            return page.extract_text() or ""

        page_width = page.width
        midpoint = page_width / 2

        # Cluster words into lines by vertical position (tolerant of small
        # baseline jitter between adjacent words on the same line).
        lines = self._cluster_into_lines(words)

        left_words_total = sum(1 for w in words if w["x1"] < midpoint - 5)
        right_words_total = sum(1 for w in words if w["x0"] > midpoint + 5)
        is_two_column = (
            left_words_total > 15
            and right_words_total > 15
            and left_words_total + right_words_total > 0.7 * len(words)
        )

        if not is_two_column:
            return page.extract_text() or ""

        # Classify each line as "spanning" (crosses the midpoint / covers
        # most of the page width -> likely a header, footer, or full-width
        # section divider) or as belonging to the left/right column.
        output_segments: List[str] = []
        left_buffer: List[str] = []
        right_buffer: List[str] = []

        def flush_columns():
            if left_buffer:
                output_segments.append("\n".join(left_buffer))
                left_buffer.clear()
            if right_buffer:
                output_segments.append("\n".join(right_buffer))
                right_buffer.clear()

        for line_words in lines:
            line_x0 = min(w["x0"] for w in line_words)
            line_x1 = max(w["x1"] for w in line_words)
            line_text = " ".join(w["text"] for w in line_words)
            spans_both_sides = line_x0 < midpoint - 10 and line_x1 > midpoint + 10
            covers_most_width = (line_x1 - line_x0) > 0.6 * page_width

            if spans_both_sides or covers_most_width:
                flush_columns()
                output_segments.append(line_text)
            else:
                center = (line_x0 + line_x1) / 2
                if center < midpoint:
                    left_buffer.append(line_text)
                else:
                    right_buffer.append(line_text)

        flush_columns()
        return "\n".join(output_segments)

    @staticmethod
    def _cluster_into_lines(
        words, y_tolerance: float = 3.0, x_gap_ratio: float = 0.12
    ) -> List[list]:
        """
        Group words into visual lines using BOTH vertical proximity and
        horizontal continuity. Two-column layouts frequently place a line
        of the left column at the exact same y-coordinate as a line of the
        right column (matching line spacing) -- grouping by y alone would
        wrongly merge them into a single scrambled line. A large x-gap
        between consecutive words (bigger than a fraction of page width,
        i.e. roughly the empty gutter between columns) forces a line break
        even when y matches.
        """
        if not words:
            return []

        page_width = max(w["x1"] for w in words) or 1.0
        min_gap = max(40.0, x_gap_ratio * page_width)

        sorted_words = sorted(words, key=lambda w: (round(w["top"] / y_tolerance), w["x0"]))
        lines: List[list] = []
        current_line: list = []
        current_top = None
        current_x1 = None

        for w in sorted_words:
            same_row = current_top is not None and abs(w["top"] - current_top) <= y_tolerance
            contiguous = current_x1 is not None and (w["x0"] - current_x1) <= min_gap

            if same_row and contiguous:
                current_line.append(w)
                current_x1 = w["x1"]
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [w]
                current_top = w["top"]
                current_x1 = w["x1"]

        if current_line:
            lines.append(current_line)

        for line in lines:
            line.sort(key=lambda w: w["x0"])

        return lines

    def _guess_column_count(self, page) -> int:
        words = page.extract_words()
        if not words:
            return 1
        midpoint = page.width / 2
        left = sum(1 for w in words if w["x1"] < midpoint - 5)
        right = sum(1 for w in words if w["x0"] > midpoint + 5)
        if left > 15 and right > 15:
            return 2
        return 1

    def _ocr_page(self, page) -> str:
        if not _HAS_OCR:
            return ""
        try:
            im = page.to_image(resolution=200).original
            return pytesseract.image_to_string(im)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR failed: %s", exc)
            return ""

    # ------------------------------------------------------------------ #
    # PyPDF2 fallback path (text only, no table/column awareness)
    # ------------------------------------------------------------------ #
    def _read_with_pypdf2(self, path: Path) -> PDFReadResult:
        pages: List[PageResult] = []
        warnings: List[str] = ["Using PyPDF2 fallback: no table/column-aware extraction."]

        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    warnings.append(f"Page {i}: no extractable text found.")
                pages.append(PageResult(page_number=i, text=text))

        return PDFReadResult(
            source_path=str(path), pages=pages, engine_used="PyPDF2", warnings=warnings
        )
