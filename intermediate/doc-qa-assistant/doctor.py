"""Route doctor for DocuAsk.

    python doctor.py           # show the resolved configuration
    python doctor.py --live    # configuration + one real grounded request
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def masked(value: str | None) -> str:
    if not value:
        return "<missing>"
    return value[:6] + "..." + value[-4:] if len(value) > 12 else "<set>"


REQUIRED_MODULES = ("streamlit", "requests", "pypdf", "dotenv")


def dependency_report(modules) -> str:
    """Report which key third-party packages are importable in this interpreter."""

    parts = []
    for module in modules:
        try:
            __import__(module)
            parts.append(f"{module}=ok")
        except Exception:
            parts.append(f"{module}=MISSING")
    return ", ".join(parts)


def main() -> int:
    from config import load_settings
    from ingest import Document
    from llm_client import LLMError
    from qa import ask, build_index

    settings = load_settings()
    print("Resolved configuration")
    print(f"  config file      : {os.path.join(HERE, '.env')}  (exists: {os.path.exists(os.path.join(HERE, '.env'))})")
    print(f"  LLM_PROVIDER     : {settings.llm_provider}")
    print(f"  base_url         : {settings.llm_base_url or '<none>'}")
    print(f"  model            : {settings.llm_model}")
    print(f"  credential       : {masked(settings.llm_api_key)}")
    print(f"  embeddings       : {settings.embedding_provider}")
    print(f"  dependencies     : {dependency_report(REQUIRED_MODULES)}")

    if "--live" not in sys.argv:
        print("\nAdd --live to send one real request through this route.")
        return 0

    print("\nBuilding a tiny index and asking one live question ...")
    docs = [Document("doctor.txt", "doctor.txt", "A Zorblax reading above 40 indicates the Orbital Loom needs recalibration.", "text")]
    index = build_index(docs, settings)
    try:
        result = ask(index, "What does a Zorblax reading above 40 indicate?")
    except LLMError as exc:
        print(f"  FAILED ({exc.kind}): {exc.message}")
        return 1

    print(f"  route used       : {result.provider}")
    print(f"  refused          : {result.refused}")
    print(f"  groundedness     : {result.groundedness}")
    print(f"  sources          : {[s.citation for s in result.sources]}")
    print("  answer head      : " + result.answer.strip().replace("\n", " ")[:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
