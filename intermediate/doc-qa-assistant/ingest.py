"""Document ingestion: TXT / Markdown / PDF -> plain text.

Only ``pypdf`` is optional (needed for PDF). Everything else uses the standard
library. Invalid files raise :class:`IngestError` with a user-friendly message.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


class IngestError(Exception):
    def __init__(self, message: str, *, kind: str = "ingest") -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind


@dataclass
class Document:
    doc_id: str
    name: str
    text: str
    source_type: str
    meta: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


_TEXT_EXTS = {".txt", ".md", ".markdown"}
_PDF_EXTS = {".pdf"}


def _read_text_file(path: str) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as handle:
                return handle.read()
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            raise IngestError(f"Could not read '{os.path.basename(path)}': {exc}", kind="io") from exc
    raise IngestError(f"Could not decode '{os.path.basename(path)}' as text.", kind="encoding")


def _read_pdf_file(path: str) -> str:
    try:
        import pypdf
    except Exception as exc:  # pragma: no cover
        raise IngestError(
            "PDF support requires the 'pypdf' package. Install it with `pip install pypdf`.",
            kind="dependency",
        ) from exc

    try:
        reader = pypdf.PdfReader(path)
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:
        raise IngestError(f"Could not parse PDF '{os.path.basename(path)}': {exc}", kind="pdf") from exc

    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text.strip():
        raise IngestError(
            f"'{os.path.basename(path)}' appears to be empty or image-only (no extractable text).",
            kind="empty",
        )
    return text


def load_document(path: str, max_file_mb: float = 10.0) -> Document:
    """Load a single document from disk into a :class:`Document`."""

    if not os.path.exists(path):
        raise IngestError(f"File not found: {path}", kind="missing")
    if os.path.isdir(path):
        raise IngestError(f"'{os.path.basename(path)}' is a directory, not a file.", kind="invalid")

    ext = os.path.splitext(path)[1].lower()
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > max_file_mb:
        raise IngestError(
            f"'{os.path.basename(path)}' is {size_mb:.1f} MB which exceeds the {max_file_mb} MB limit.",
            kind="too_large",
        )

    if ext in _TEXT_EXTS:
        text = _read_text_file(path)
        source_type = "markdown" if ext in (".md", ".markdown") else "text"
    elif ext in _PDF_EXTS:
        text = _read_pdf_file(path)
        source_type = "pdf"
    else:
        raise IngestError(
            f"Unsupported file type '{ext or 'unknown'}'. Supported: .txt, .md, .pdf.",
            kind="unsupported",
        )

    if not text.strip():
        raise IngestError(f"'{os.path.basename(path)}' contains no readable text.", kind="empty")

    name = os.path.basename(path)
    return Document(
        doc_id=name,
        name=name,
        text=text,
        source_type=source_type,
        meta={"size_kb": round(os.path.getsize(path) / 1024, 1), "chars": len(text)},
    )


def load_bytes(name: str, data: bytes, max_file_mb: float = 10.0) -> Document:
    """Load a document from an in-memory byte payload (used by the web UI)."""

    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_file_mb:
        raise IngestError(f"'{name}' is {size_mb:.1f} MB which exceeds the {max_file_mb} MB limit.", kind="too_large")

    ext = os.path.splitext(name)[1].lower()
    if ext in _TEXT_EXTS:
        text = data.decode("utf-8", errors="replace")
        source_type = "markdown" if ext in (".md", ".markdown") else "text"
    elif ext in _PDF_EXTS:
        try:
            import io

            import pypdf
        except Exception as exc:  # pragma: no cover
            raise IngestError("PDF support requires the 'pypdf' package.", kind="dependency") from exc
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            text = "\n\n".join((page.extract_text() or "").strip() for page in reader.pages)
        except Exception as exc:
            raise IngestError(f"Could not parse PDF '{name}': {exc}", kind="pdf") from exc
        source_type = "pdf"
    else:
        raise IngestError(f"Unsupported file type '{ext or 'unknown'}'. Supported: .txt, .md, .pdf.", kind="unsupported")

    if not text.strip():
        raise IngestError(f"'{name}' contains no readable text.", kind="empty")
    return Document(doc_id=name, name=name, text=text, source_type=source_type, meta={"size_kb": round(size_mb * 1024, 1)})
