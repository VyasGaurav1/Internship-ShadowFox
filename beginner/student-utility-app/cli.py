"""Command line runner for the Student Utility App.

Lets the whole workflow be exercised without a browser:

    python cli.py --list-features
    python cli.py --feature summarize --input-file notes.txt
    python cli.py --feature quiz --input "..." --num-questions 4
    python cli.py --feature improve --input "..." --topic "photosynthesis"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import run_feature
from prompts import FEATURES, FEATURE_LABELS


def _read_input(args: argparse.Namespace) -> str:
    if args.input_file:
        return Path(args.input_file).read_text(encoding="utf-8")
    return args.input or ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="student-utility-app",
        description="AI-powered study helper: summarise notes, build quizzes, improve answers, explain concepts.",
    )
    parser.add_argument("--feature", choices=FEATURES, help="utility feature to run")
    parser.add_argument("--input", help="text content (notes, answer, material)")
    parser.add_argument("--input-file", help="read the content from a UTF-8 text file instead")
    parser.add_argument("--topic", default="", help="topic/question (used by improve/explain)")
    parser.add_argument("--num-questions", type=int, default=5, help="questions for the quiz feature")
    parser.add_argument("--list-features", action="store_true", help="list available features and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_features:
        print("Available features:")
        for key in FEATURES:
            print(f"  {key:<10} {FEATURE_LABELS[key]}")
        return 0

    if not args.feature:
        parser.print_help()
        return 2

    try:
        content = _read_input(args)
    except OSError as exc:
        print(f"[input error] Could not read the input file: {exc}", file=sys.stderr)
        return 2

    options = {"topic": args.topic, "num_questions": args.num_questions}
    result = run_feature(args.feature, content, options)

    if not result["ok"]:
        print(f"[{result['error_kind']}] {result['error']}", file=sys.stderr)
        return 1

    mode = "offline mode" if result["offline"] else f"{result['provider']} ({result['model']})"
    print(f"--- {FEATURE_LABELS[result['feature']]} | provider: {mode} | {result['elapsed_ms']} ms ---\n")
    print(result["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
