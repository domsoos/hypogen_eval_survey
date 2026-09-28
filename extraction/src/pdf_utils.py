from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF


def extract_pdf_pages(pdf_path: str | Path) -> List[Dict[str, object]]:
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text", sort=True).strip()
        pages.append({"page": i, "text": text})
    return pages


def format_paper_with_page_markers(pdf_path: str | Path) -> str:
    pages = extract_pdf_pages(pdf_path)
    chunks = []
    for item in pages:
        chunks.append(f"\n===== PDF PAGE {item['page']} =====\n{item['text']}\n")
    return "".join(chunks).strip()


def paper_stats(pdf_path: str | Path) -> dict:
    pages = extract_pdf_pages(pdf_path)
    text = "\n".join(str(p["text"]) for p in pages)
    return {
        "pages": len(pages),
        "characters": len(text),
        "words": len(text.split()),
    }
