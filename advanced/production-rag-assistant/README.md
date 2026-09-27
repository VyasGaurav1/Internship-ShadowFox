# Production RAG Assistant (Advanced Tier)

A production-style Retrieval-Augmented Generation system with **FastAPI +
Streamlit + Pydantic + LangGraph + FAISS + Docker**, answering questions grounded
in PDF / TXT / Markdown documents. It is built as a layered system (ingestion →
retrieval → generation) with typed contracts, guardrails, observability and a
containerised setup — not a single-file prototype.

---

## 1. Architecture

```
┌───────────────────────────── FastAPI (app/main.py) ─────────────────────────────┐
│  POST /ingest · POST /ingest/text · POST /ask · POST /ask/stream · GET /health   │
│                     GET /documents · DELETE /documents                           │
└───────────────────────────────────────┬──────────────────────────────────────────┘
                                        │ typed Pydantic schemas (app/schemas.py)
                                        ▼
                            app/service.py  (RAGService)
                                        │
        ┌───────────────────────────────┴───────────────────────────────┐
        │        LangGraph orchestration (app/graph/workflow.py)         │
        │                                                                │
        │  rewrite_query → retrieve → rerank → grade_context             │
        │        │                                  │                    │
        │        │ (weak context) ───────────────► refuse                │
        │        ▼                                  │                    │
        │     generate → verify_groundedness        │                    │
        │        │              │ (not grounded) ──► refuse              │
        │        └──────────► finish                                     │
        └────────────────────────────────────────────────────────────────┘
                                        │
   app/pipeline/: ingest · preprocess · chunking · embeddings · vector_store ·
                  retriever · reranker · generator · guardrails
```

If `langgraph` is unavailable the same node functions run through a linear
fallback (`app/graph/fallback.py`) with identical semantics; `/health` reports
which orchestration is active.

---

## 2. Why each component

| Component | Role | Why |
|---|---|---|
| **FastAPI** | HTTP API | Typed request/response, automatic 422 validation, async uploads, OpenAPI docs. |
| **Pydantic v2** | Schemas | One source of truth for validation and serialisation at every boundary. |
| **LangGraph** | Orchestration | Explicit state machine for a multi-step agentic flow, with conditional refusal branches. |
| **FAISS** (`IndexFlatIP`) | Vector index | Fast exact inner-product search over L2-normalised vectors; cosine fallback if unavailable. |
| **hybrid reranker** | Retrieval quality | Vector score + lexical overlap + MMR diversity before generation. |
| **guardrails** | Trust | Context grading pre-generation and groundedness verification post-generation. |
| **Streamlit** | UI | Upload, query, and inspect the retrieved context; can drive the API or run in-process. |
| **Docker / Compose** | Reproducibility | Identical runtime everywhere; api + ui services wired together. |

---

## 3. Pipeline detail

1. **Multi-format ingestion** (`ingest.py`) — `.txt`, `.md`, `.markdown`, `.pdf`; PDF text via `pypdf`; typed errors (`unsupported`, `too_large`, `empty`, `pdf`, …) mapped to HTTP status codes.
2. **Preprocessing** (`preprocess.py`) — Unicode NFC normalisation, hyphen line-break repair, running header/footer removal, blank-line and whitespace collapse.
3. **Chunking** (`chunking.py`) — **sentence-aware** windows (default 160 words, 40-word overlap): chunks end on sentence boundaries, keeping each embedded unit self-contained; exact character offsets are retained for citations.
4. **Embeddings** (`embeddings.py`) — offline deterministic **hashing** embedder by default; optional `sentence-transformers` or OpenAI embeddings.
5. **Vector indexing** (`vector_store.py`) — FAISS when available, otherwise a dependency-free cosine store; same interface.
6. **Document-scoped retrieval** (`retriever.py`) — optional `document_ids` filter so a question can be restricted to specific files.
7. **Reranking** (`reranker.py`) — hybrid relevance (vector + lexical) followed by MMR diversity selection.
8. **Grounded generation** (`generator.py`) — LLM conditioned strictly on numbered passages (must cite `[n]`, must refuse if insufficient); deterministic **extractive** answerer when offline.
9. **Guardrails** (`guardrails.py`) — refuse when retrieval is weak, and refuse when the generated answer's groundedness is below `MIN_GROUNDEDNESS`.

### How hallucinations are reduced
* Retrieval precedes generation; the prompt contains only retrieved passages.
* The model must cite passage numbers and is forbidden from using outside knowledge.
* Two independent refusal gates (weak-context and low-groundedness).
* Every answer exposes its sources, similarity scores, rerank scores, groundedness and pipeline steps.

---

## 4. How to run

### Local (Python)
```bash
cd advanced/production-rag-assistant
python -m venv .venv
.venv\Scripts\activate                 # Windows
pip install -r requirements.txt        # faiss-cpu is optional; cosine fallback works without it

python scripts/make_sample_pdf.py sample_docs/alpha9_handbook.pdf   # create the demo PDF

# API
uvicorn app.main:app --reload --port 8000
#   docs at http://localhost:8000/docs

# UI (in a second terminal)
streamlit run ui/streamlit_app.py
```

### Docker
```bash
docker compose up --build
# API  -> http://localhost:8000/docs
# UI   -> http://localhost:8501
```

### Verify offline (no key, no network)
```bash
python selfcheck.py     # 12 end-to-end checks against the real FastAPI app
pytest -q
```

---

## 5. API examples

```bash
# ingest files
curl -X POST http://localhost:8000/ingest -F "files=@sample_docs/alpha9_notes.txt"

# grounded question
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"What does a Zorblax reading above 40 indicate?","top_k":4}'

# document-scoped question
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"...","document_ids":["alpha9_notes.txt"]}'

# streaming (SSE)
curl -N -X POST http://localhost:8000/ask/stream -H "Content-Type: application/json" \
  -d '{"question":"..."}'
```

---

## 6. Project layout

```
production-rag-assistant/
├── app/
│   ├── main.py               # FastAPI app + routes
│   ├── service.py            # RAGService: pipeline steps + orchestration
│   ├── schemas.py            # Pydantic v2 request/response models
│   ├── settings.py           # typed, env-driven settings
│   ├── graph/
│   │   ├── workflow.py       # LangGraph StateGraph
│   │   ├── fallback.py       # identical linear orchestration
│   │   └── state.py          # shared TypedDict state
│   └── pipeline/
│       ├── ingest.py preprocess.py chunking.py embeddings.py
│       ├── vector_store.py retriever.py reranker.py
│       └── generator.py guardrails.py text_utils.py
├── ui/streamlit_app.py       # Streamlit front-end (in-process or API mode)
├── scripts/make_sample_pdf.py
├── sample_docs/              # alpha9_notes.txt, photosynthesis.md, alpha9_handbook.pdf
├── tests/                    # pytest: API + pipeline
├── selfcheck.py              # end-to-end offline verification
├── Dockerfile · docker-compose.yml · .dockerignore
├── requirements.txt · .env.example · README.md
```

---

## 7. Verified behaviour

| Command (offline) | Expected |
|---|---|
| `python selfcheck.py` | `12/12 checks passed`, exit `0` |
| `pytest -q` | all tests pass |
| `curl /health` | `orchestration: langgraph`, store `faiss-flatip` (when FAISS installed) |

The self-check drives the real FastAPI app with `TestClient`: ingestion of
TXT/MD/PDF, grounded answering, reranking, document-scoped retrieval, refusal,
typed validation (422), streaming SSE, and index clearing.

---

## 8. Live mode (OpenRouter)

Answers are generated by **OpenRouter** when a key is configured:

```ini
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1   # or your own compatible gateway
LLM_MODEL=openai/gpt-4o-mini
```

The graph's `generate` node sends the number-passaged retrieved context and requires
citations; the `verify_groundedness` node rejects answers below `MIN_GROUNDEDNESS`.
If the live model is unreachable, generation falls back to the deterministic
extractive answer (still grounded, still cited) and the response carries a warning.
Reasoning models that return `content: null` with a `reasoning` field are handled.
`.env` is gitignored.
