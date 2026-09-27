"""Multi-format document ingestion: .txt, .md, .markdown, .pdf.

Returns a typed :class:`RawDocument`; all failure modes raise
:class:`IngestError` with a stable ``kind`` so the API can map them to HTTP
status codes.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field

_TEXT_EXTS = {".txt", ".md", ".markdown"}
_PDF_EXTS = {".pdf"}


class IngestError(Exception):
    def __init__(self, message: str, *, kind: str = "ingest", status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.status = status


@dataclass
class RawDocument:
    doc_id: str
    filename: str
    source_type: str
    text: str
    meta: dict = field(default_factory=dict)

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def words(self) -> int:
        return len(self.text.split())


def _ext_of(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()


def _check_ext(filename: str, allowed: tuple[str, ...]) -> str:
    ext = _ext_of(filename)
    if ext not in allowed:
        raise IngestError(
            f"Unsupported file type '{ext or 'unknown'}' for '{filename}'. Supported: {', '.join(allowed)}.",
            kind="unsupported",
            status=415,
        )
    return ext


def _decode_text(data: bytes, filename: str) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise IngestError(f"Could not decode '{filename}' as text.", kind="encoding", status=422)


def _extract_pdf(data: bytes, filename: str) -> tuple[str, int]:
    try:
        import pypdf
    except Exception as exc:  # pragma: no cover
        raise IngestError("PDF support requires the 'pypdf' package.", kind="dependency", status=500) from exc
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:
        raise IngestError(f"Could not parse PDF '{filename}': {exc}", kind="pdf", status=422) from exc
    text = "\n\n".join(p for p in pages if p)
    if not text.strip():
        raise IngestError(
            f"'{filename}' has no extractable text (it may be scanned/image-only).", kind="empty", status=422
        )
    return text, len(pages)


def load_bytes(filename: str, data: bytes, *, max_file_mb: float = 12.0, allowed: tuple[str, ...] = (".txt", ".md", ".markdown", ".pdf")) -> RawDocument:
    """Ingest an in-memory file payload (used by the API upload endpoint)."""

    if not filename:
        raise IngestError("A filename is required.", kind="invalid", status=400)
    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_file_mb:
        raise IngestError(f"'{filename}' is {size_mb:.1f} MB which exceeds the {max_file_mb} MB limit.", kind="too_large", status=413)
    if not data:
        raise IngestError(f"'{filename}' is empty.", kind="empty", status=422)

    ext = _check_ext(filename, allowed)
    if ext in _PDF_EXTS:
        text, pages = _extract_pdf(data, filename)
        return RawDocument(doc_id=filename, filename=filename, source_type="pdf", text=text, meta={"pages": pages})
    text = _decode_text(data, filename)
    if not text.strip():
        raise IngestError(f"'{filename}' contains no readable text.", kind="empty", status=422)
    source_type = "markdown" if ext in (".md", ".markdown") else "text"
    return RawDocument(doc_id=filename, filename=filename, source_type=source_type, text=text, meta={})


def load_path(path: str, *, max_file_mb: float = 12.0, allowed: tuple[str, ...] = (".txt", ".md", ".markdown", ".pdf")) -> RawDocument:
    """Ingest a file from disk."""

    if not os.path.exists(path):
        raise IngestError(f"File not found: {path}", kind="missing", status=404)
    if os.path.isdir(path):
        raise IngestError(f"'{os.path.basename(path)}' is a directory.", kind="invalid", status=400)
    with open(path, "rb") as handle:
        data = handle.read()
    return load_bytes(os.path.basename(path), data, max_file_mb=max_file_mb, allowed=allowed)
