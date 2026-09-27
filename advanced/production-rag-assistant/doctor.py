"""Route doctor for the Production RAG Assistant.

    python doctor.py           # show the resolved configuration
    python doctor.py --live    # configuration + one real grounded request through the graph
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


REQUIRED_MODULES = (
    "fastapi", "pydantic", "uvicorn", "multipart", "pypdf", "requests",
    "langgraph", "faiss", "numpy", "streamlit", "dotenv",
)


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
    from app.service import RAGService

    service = RAGService()
    settings = service.settings
    print("Resolved configuration")
    print(f"  config file      : {os.path.join(HERE, '.env')}  (exists: {os.path.exists(os.path.join(HERE, '.env'))})")
    print(f"  LLM_PROVIDER     : {settings.llm_provider}")
    print(f"  base_url         : {settings.llm_base_url or '<none>'}")
    print(f"  model            : {settings.llm_model}")
    print(f"  credential       : {masked(settings.llm_api_key)}")
    print(f"  embeddings       : {settings.embedding_provider}")
    print(f"  orchestration    : {service.pipeline().mode}")
    print(f"  dependencies     : {dependency_report(REQUIRED_MODULES)}")

    if "--live" not in sys.argv:
        print("\nAdd --live to send one real request through this route.")
        return 0

    print("\nIngesting a tiny document and asking one live question ...")
    service.ingest_bytes(
        [("doctor.txt", b"A Zorblax reading above 40 indicates the Orbital Loom needs recalibration.")]
    )
    result = service.ask("What does a Zorblax reading above 40 indicate?")
    print(f"  route used       : {result['provider']}")
    print(f"  refused          : {result['refused']}")
    print(f"  groundedness     : {result['groundedness']}")
    print(f"  steps            : {' -> '.join(result['steps'])}")
    print(f"  sources          : {[c.chunk.citation for c in result['contexts']]}")
    print("  answer head      : " + result["answer"].strip().replace("\n", " ")[:200])
    return 0 if not result["refused"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
