"""Generate a minimal but valid single-page PDF for the demo documents.

Uses only the standard library: it writes a correct PDF 1.4 file with proper
xref offsets, so `pypdf` can read it and extract the text. Run:

    python tools/make_sample_pdf.py sample_docs/sample.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

LINES = [
    "Alpha-9 Experiment - Calibration Notes",
    "",
    "Zorblax is a fictional calibration unit used only in the Alpha-9 experiment.",
    "The Alpha-9 experiment measures thermal drift in the Orbital Loom device.",
    "A Zorblax reading above 40 indicates the Orbital Loom needs recalibration.",
    "Calibration is performed by the Qarn tool every 500 cycles.",
    "",
    "Safety: never exceed a Zorblax reading of 65 during live operation.",
]


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf_bytes(lines: list[str]) -> bytes:
    parts = ["BT", "/F1 11 Tf", "54 742 Td"]
    for index, line in enumerate(lines):
        if index:
            parts.append("0 -18 Td")
        parts.append(f"({_escape(line)}) Tj")
    parts.append("ET")
    content = "\n".join(parts).encode("latin-1", "replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    )
    return bytes(out)


def main(argv: list[str]) -> int:
    target = Path(argv[1] if len(argv) > 1 else "sample_docs/sample.pdf")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(build_pdf_bytes(LINES))
    print(f"Wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
