"""Streamlit UI for the Document Q&A Assistant.

    streamlit run app.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    import streamlit as st

    from config import load_settings, with_overrides
    from ingest import IngestError, load_bytes
    from qa import ValidationError, ask, build_index

    st.set_page_config(page_title="DocuAsk - Document Q&A", page_icon="📄", layout="wide")
    settings = load_settings()

    st.title("📄 DocuAsk")
    st.caption("Upload documents, then ask questions answered **only** from their content - with the retrieved sources shown.")

    if settings.offline_llm:
        st.info(
            "Running in **offline mode**: answers are composed deterministically from retrieved passages, "
            "so grounding is verifiable without any API key. Configure an LLM in `.env` for fluent generation."
        )

    # ----------------------------- sidebar ----------------------------- #
    with st.sidebar:
        st.header("Ingestion settings")
        chunk_size = st.slider("Chunk size (words)", 60, 400, settings.chunk_size, step=20)
        overlap = st.slider("Chunk overlap (words)", 0, 120, settings.chunk_overlap, step=10)
        top_k = st.slider("Chunks retrieved per question", 1, 8, settings.top_k)

        st.divider()
        st.header("Documents")
        uploads = st.file_uploader(
            "Upload .txt / .md / .pdf",
            type=["txt", "md", "markdown", "pdf"],
            accept_multiple_files=True,
        )

        index = st.session_state.get("index")
        if index is not None:
            st.success(f"Index ready: {index.stats['documents']} docs · {index.stats['chunks']} chunks")
            st.caption(
                f"chunk size {index.stats['chunk_size']} · overlap {index.stats['chunk_overlap']} · "
                f"embedder {index.stats['embedding_provider']} · store {index.stats['vector_store']}"
            )
            if st.button("Clear index"):
                st.session_state["index"] = None
                st.rerun()

        build = st.button("Build / rebuild index", type="primary", use_container_width=True)

    if build:
        if not uploads:
            st.error("Please upload at least one document first.")
        else:
            documents, problems = [], []
            with st.spinner("Ingesting, chunking and embedding..."):
                for upload in uploads:
                    try:
                        documents.append(load_bytes(upload.name, upload.getvalue(), max_file_mb=settings.max_file_mb))
                    except IngestError as exc:
                        problems.append(f"{upload.name}: {exc.message}")
            if documents:
                try:
                    local_settings = with_overrides(
                        settings, chunk_size=chunk_size, chunk_overlap=overlap
                    )
                    st.session_state["index"] = build_index(documents, local_settings)
                    st.session_state["top_k"] = top_k
                except IngestError as exc:
                    problems.append(exc.message)
            for problem in problems:
                st.warning(problem)

    index = st.session_state.get("index")
    if index is None:
        st.subheader("1. Upload documents")
        st.markdown("Upload one or more `.txt`, `.md` or `.pdf` files in the sidebar, then press **Build / rebuild index**.")
        st.stop()

    st.subheader("2. Ask a question")
    question = st.text_input("Your question", placeholder="e.g. What is a list comprehension?")
    if st.button("Ask", type="primary") or question:
        try:
            validated = question.strip()
            if not validated:
                st.warning("Please type a question first.")
                st.stop()
            effective_top_k = st.session_state.get("top_k", settings.top_k)
            if effective_top_k != index.settings.top_k:
                index.settings = with_overrides(index.settings, top_k=effective_top_k)
            with st.spinner("Searching the documents..."):
                result = ask(index, validated)
        except ValidationError as exc:
            st.error(exc.message)
            st.stop()

        st.subheader("Answer")
        if result.refused:
            st.warning(result.answer)
        else:
            st.markdown(result.answer)
            st.caption(
                f"provider: {result.provider} · groundedness: {result.groundedness:.0%} · {result.elapsed_ms} ms"
                + (f" · ⚠ {result.warning}" if result.warning else "")
            )

        if result.sources:
            with st.expander("Retrieved sources (supporting text)", expanded=False):
                for source in result.sources:
                    st.markdown(f"**{source.citation}** · similarity {source.score}")
                    st.text(source.text[:900] + ("..." if len(source.text) > 900 else ""))
                    st.divider()

    with st.expander("How grounding works"):
        st.markdown(
            "1. **Ingest** - text is extracted from TXT/MD/PDF.\n"
            "2. **Chunk** - documents are split into overlapping word windows.\n"
            "3. **Embed** - every chunk becomes a vector (offline hashing embedder by default).\n"
            "4. **Retrieve** - the question is embedded and the most similar chunks are selected.\n"
            "5. **Ground** - the answer is generated *only* from those chunks and must cite them.\n"
            "6. **Refuse** - if nothing relevant is retrieved (score below the threshold), the assistant says so "
            "instead of guessing."
        )


if __name__ == "__main__":
    main()
