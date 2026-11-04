"""
src/loader.py
PDF loading and text extraction.

read_pdf() returns a list of (page_number, text) tuples, which is exactly
what chunk_pages() in chunking.py expects.
"""

import re
from pathlib import Path

try:
    import pymupdf as fitz  # PyMuPDF (new name)
except ImportError:
    import fitz  # PyMuPDF (older versions)


def clean_text(text):
    """Light cleanup of text pulled out of a PDF."""
    # Join words that were split across lines with a hyphen: "retrie-\nval" -> "retrieval"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse runs of spaces/tabs (newlines are handled later by the chunker)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def read_pdf(path):
    """Return a list of (page_number, text) for every page that has text."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {path}")

    pages = []
    with fitz.open(path) as doc:
        for number, page in enumerate(doc, start=1):
            text = clean_text(page.get_text())
            if text:
                pages.append((number, text))

    # TODO: scanned PDFs have no text layer. Add OCR (for example pytesseract)
    # here when `pages` is empty.
    if not pages:
        raise ValueError(f"No extractable text found in {path}. It may be a scanned PDF.")
    return pages


def list_pdfs(folder="data"):
    """Return all PDF files in a folder (useful for the data/ sample PDFs)."""
    return sorted(Path(folder).glob("*.pdf"))
