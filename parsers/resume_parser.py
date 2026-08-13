"""
parsers/resume_parser.py
-------------------------
Extracts raw text and basic structured fields (email, phone) from
resume files (PDF / DOCX). This is a starting scaffold — extend
`extract_text_from_pdf` / `extract_text_from_docx` with real parsing
logic (pdfplumber / python-docx) as the project matures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from utils.logger import get_logger
from utils.validators import is_valid_email, is_valid_phone

logger = get_logger("parsers.resume_parser")


@dataclass
class ParsedResume:
    source_file: str
    raw_text: str
    email: str | None = None
    phone: str | None = None


def extract_text_from_pdf(filepath: Path) -> str:
    """Extract raw text from a PDF resume using pdfplumber."""
    import pdfplumber

    logger.info("Extracting text from PDF: %s", filepath)
    text_chunks: list[str] = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text_chunks.append(page.extract_text() or "")
    return "\n".join(text_chunks)


def extract_text_from_docx(filepath: Path) -> str:
    """Extract raw text from a DOCX resume using python-docx."""
    import docx

    logger.info("Extracting text from DOCX: %s", filepath)
    document = docx.Document(str(filepath))
    return "\n".join(p.text for p in document.paragraphs)


def parse_resume(filepath: str | Path) -> ParsedResume:
    """
    Parse a resume file (PDF or DOCX) into a ParsedResume object.
    Raises ValueError for unsupported file types.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == ".pdf":
        raw_text = extract_text_from_pdf(filepath)
    elif suffix == ".docx":
        raw_text = extract_text_from_docx(filepath)
    else:
        logger.error("Unsupported resume file type: %s", suffix)
        raise ValueError(f"Unsupported file type: {suffix}")

    email = next((w.strip(".,;") for w in raw_text.split() if is_valid_email(w.strip(".,;"))), None)
    phone_candidate = _find_phone_candidate(raw_text)

    parsed = ParsedResume(
        source_file=str(filepath),
        raw_text=raw_text,
        email=email,
        phone=phone_candidate,
    )
    logger.info("Parsed resume %s (email found: %s)", filepath.name, bool(email))
    return parsed


def _find_phone_candidate(text: str) -> str | None:
    for line in text.splitlines():
        candidate = line.strip()
        if is_valid_phone(candidate):
            return candidate
    return None
