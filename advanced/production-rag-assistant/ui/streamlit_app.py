"""Streamlit UI for the Production RAG Assistant.

Talks to the in-process RAG service by default. If ``RAG_API_URL`` is set it
calls the FastAPI backend instead, so the UI works both as a standalone app and
as a thin client over the API.

    streamlit run ui/streamlit_app.py
"""
from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


def _get_service(st):
    if "service" not in st.session_state:
        from app.service import RAGService

        st.session_state["service"] = RAGService()
    return st.session_state["service"]


def _api_mode() -> str | None:
    return os.getenv("RAG_API_URL") or None


def main() -> None:
    import requests
    import streamlit as st

    st.set_page_config(page_title="Production RAG Assistant", page_icon="🧠", layout="wide")
    st.title("🧠 Production RAG Assistant")
    st.caption("Multi-format ingestion · FAISS/hybrid retrieval · reranking · grounded answers with citations.")

    api_url = _api_mode()
    if api_url:
        st.info(f"API mode: calling the FastAPI backend at {api_url}")
        try:
            stats = requests.get(f"{api_url}/health", timeout=5).json()
        except Exception as exc:
            st.error(f"Could not reach the API at {api_url}: {exc}")
            st.stop()
    else:
        service = _get_service(st)
        stats = service.stats

    with st.sidebar:
        st.header("Index status")
        st.metric("Documents", stats.get("documents", 0))
        st.metric("Chunks", stats.get("chunks", 0))
        st.caption(
            f"embedder: {stats.get('embedding_provider')} · store: {stats.get('vector_store')}\n\n"
            f"llm: {stats.get('llm_provider')} · orchestration: {stats.get('orchestration')}"
        )
        if not api_url and st.button("Clear index"):
            service.clear()
            st.rerun()

        st.divider()
        st.header("Documents")
        uploads = st.file_uploader("Upload .txt / .md / .pdf", type=["txt", "md", "markdown", "pdf"], accept_multiple_files=True)
        if st.button("Ingest", type="primary", use_container_width=True):
            if not uploads:
                st.error("Please choose at least one file.")
            elif api_url:
                files = [("files", (up.name, up.getvalue())) for up in uploads]
                response = requests.post(f"{api_url}/ingest", files=files, timeout=120)
                st.success(response.json()) if response.ok else st.error(response.text)
            else:
                try:
                    outcome = service.ingest_bytes([(up.name, up.getvalue()) for up in uploads])
                    st.success(f"Ingested {len(outcome.documents)} document(s) · {outcome.total_chunks} chunks total")
                except Exception as exc:  # IngestError or unexpected
                    st.error(getattr(exc, "message", str(exc)))

        st.divider()
        st.header("Retrieval options")
        top_k = st.slider("Top-k context chunks", 1, 10, 4)
        rewrite = st.checkbox("Query rewriting", value=True)
        rerank_flag = st.checkbox("Hybrid reranking", value=True)

    st.subheader("Ask a question")
    question = st.text_input("Question", placeholder="e.g. What does a Zorblax reading above 40 indicate?")
    if st.button("Ask", type="primary") or (question and question.strip()):
        if not question.strip():
            st.warning("Please type a question.")
            st.stop()
        with st.spinner("Running the RAG pipeline..."):
            if api_url:
                payload = {"question": question, "top_k": top_k, "rewrite": rewrite, "rerank": rerank_flag}
                response = requests.post(f"{api_url}/ask", json=payload, timeout=120)
                if not response.ok:
                    st.error(response.text)
                    st.stop()
                result = response.json()
                answer = result["answer"]
                sources = result["sources"]
                grounded = result["groundedness"]
                steps = result["steps"]
                refused = result["refused"]
            else:
                try:
                    raw = service.ask(question, top_k=top_k, rewrite=rewrite, rerank=rerank_flag)
                except Exception as exc:
                    st.error(getattr(exc, "message", str(exc)))
                    st.stop()
                answer, sources, grounded, steps, refused = (
                    raw["answer"],
                    [s for s in raw["contexts"]],
                    raw["groundedness"],
                    raw["steps"],
                    raw["refused"],
                )

        st.subheader("Answer")
        if refused:
            st.warning(answer)
        else:
            st.markdown(answer)
            st.caption(f"groundedness: {grounded:.0%} · pipeline: {' → '.join(steps)}")

        if sources:
            with st.expander("Sources / retrieved context", expanded=True):
                for index, source in enumerate(sources, start=1):
                    if hasattr(source, "chunk"):  # in-process object
                        chunk = source.chunk
                        st.markdown(f"**[{index}] {chunk.citation}** · score {source.score}")
                        st.text(chunk.text[:800])
                    else:  # API dict
                        st.markdown(f"**[{index}] {source['citation']}** · score {source['score']}")
                        st.text(source["text"][:800])
                    st.divider()

    with st.expander("Architecture"):
        st.markdown(
            "**Pipeline (LangGraph):** rewrite → retrieve → rerank → grade_context → generate → verify_groundedness\n\n"
            "**Guardrails:** weak retrieval is refused; answers below the groundedness threshold are refused.\n\n"
            "**Components:** multi-format ingestion → preprocessing → sentence-aware chunking → embeddings "
            "(hashing / sentence-transformers / OpenAI) → FAISS or cosine store → hybrid retrieval + MMR rerank → "
            "grounded generation (LLM or deterministic extractive)."
        )


if __name__ == "__main__":
    main()
