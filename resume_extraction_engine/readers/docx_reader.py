"""
DOCX Reader
-----------
Extracts raw text from Word resumes, including:
  - body paragraphs (in document order)
  - tables (common for skills grids / 2-column contact blocks)
  - text inside text boxes / headers where python-docx can reach it
  - flags embedded images (not converted to text)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger("resume_engine.docx_reader")

try:
    import docx  # python-docx
    from docx.document import Document as _DocxDocument
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    _HAS_DOCX = True
except ImportError:  # pragma: no cover
    _HAS_DOCX = False


@dataclass
class DOCXReadResult:
    source_path: str
    text: str
    tables: List[List[List[str]]] = field(default_factory=list)
    image_count: int = 0
    warnings: List[str] = field(default_factory=list)
    engine_used: str = "python-docx"

    @property
    def raw_text(self) -> str:
        chunks = [self.text] if self.text.strip() else []
        for table in self.tables:
            for row in table:
                chunks.append(" | ".join(c or "" for c in row))
        return "\n".join(chunks)


class DOCXReader:
    """Reads DOCX resumes, preserving document order of paragraphs & tables."""

    def read(self, path: str | Path) -> DOCXReadResult:
        if not _HAS_DOCX:
            raise RuntimeError("python-docx is not installed.")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DOCX not found: {path}")

        document = docx.Document(str(path))
        warnings: List[str] = []

        body_lines: List[str] = []
        tables: List[List[List[str]]] = []

        for block in self._iter_block_items(document):
            if isinstance(block, Paragraph):
                if block.text.strip():
                    body_lines.append(block.text)
            elif isinstance(block, Table):
                table_data = [[cell.text.strip() for cell in row.cells] for row in block.rows]
                tables.append(table_data)

        image_count = self._count_images(document)
        if image_count:
            warnings.append(
                f"{image_count} embedded image(s) detected; image content not extracted as text."
            )

        text = "\n".join(body_lines)
        if not text.strip() and not tables:
            warnings.append("No extractable text or tables found in document body.")

        return DOCXReadResult(
            source_path=str(path),
            text=text,
            tables=tables,
            image_count=image_count,
            warnings=warnings,
        )

    @staticmethod
    def _iter_block_items(document: "_DocxDocument"):
        """
        Yield paragraphs and tables in the order they appear in the document
        body. python-docx doesn't expose this directly, so we walk the XML.
        """
        parent_elm = document.element.body
        for child in parent_elm.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, document)
            elif isinstance(child, CT_Tbl):
                yield Table(child, document)

    @staticmethod
    def _count_images(document: "_DocxDocument") -> int:
        try:
            return len([rel for rel in document.part.rels.values() if "image" in rel.reltype])
        except Exception:  # noqa: BLE001
            return 0
