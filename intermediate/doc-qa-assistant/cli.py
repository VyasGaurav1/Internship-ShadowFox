"""Command line runner for the Document Q&A Assistant.

    python cli.py ingest sample_docs/python_basics.txt
    python cli.py ask --docs sample_docs/python_basics.txt sample_docs/study_notes.md --question "..."
    python cli.py ask --docs sample_docs/python_basics.txt --question "..." --show-sources
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from pathlib import Path

from config import load_settings
from ingest import IngestError, load_document
from qa import ValidationError, ask, build_index
from validators import validate_question


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doc-qa-assistant", description="Grounded Q&A over your documents.")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="ingest documents and print index statistics")
    ingest_p.add_argument("files", nargs="+")
    ingest_p.add_argument("--chunk-size", type=int, default=None)
    ingest_p.add_argument("--overlap", type=int, default=None)

    ask_p = sub.add_parser("ask", help="ask a question grounded in the documents")
    ask_p.add_argument("--docs", nargs="+", required=True)
    ask_p.add_argument("--question", required=True)
    ask_p.add_argument("--top-k", type=int, default=None)
    ask_p.add_argument("--chunk-size", type=int, default=None)
    ask_p.add_argument("--overlap", type=int, default=None)
    ask_p.add_argument("--show-sources", action="store_true", help="print the retrieved chunk text")
    return parser


def _load_documents(paths: list[str], max_file_mb: float):
    documents = []
    for path in paths:
        documents.append(load_document(str(Path(path)), max_file_mb=max_file_mb))
    return documents


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()

    try:
        documents = _load_documents(args.docs, settings.max_file_mb)
        if args.command == "ingest":
            kwargs = {}
            if args.chunk_size:
                kwargs["chunk_size"] = args.chunk_size
            if args.overlap:
                kwargs["overlap"] = args.overlap
            index = build_index(documents, settings)
            print("Index built:")
            for key, value in index.stats.items():
                print(f"  {key:<20} {value}")
            return 0

        question = validate_question(args.question)
        index = build_index(documents, settings)
        result = ask(index, question)
    except (IngestError, ValidationError) as exc:
        print(f"[{getattr(exc, 'kind', 'error')}] {exc}", file=sys.stderr)
        return 1

    print(f"--- grounded answer | provider: {result.provider} | {result.elapsed_ms} ms ---\n")
    print(result.answer)
    if result.sources:
        print("\nSources:")
        for source in result.sources:
            print(f"  - {source.citation} (score {source.score})")
    print(f"\nGroundedness: {result.groundedness:.0%}" + ("  [REFUSED - nothing relevant found]" if result.refused else ""))

    if args.show_sources:
        print("\n=== retrieved context ===")
        for source in result.sources:
            print(f"\n--- {source.citation} ---")
            print(source.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
