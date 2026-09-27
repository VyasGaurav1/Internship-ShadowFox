# DocuAsk — Document Q&A Assistant (Intermediate Tier)

DocuAsk answers questions **grounded in your uploaded documents**, not from model
memory. It implements a full retrieval-based workflow:

```
upload -> text extraction -> chunking -> embeddings -> vector search -> grounded answer (+ sources)
```

Supported inputs: `.txt`, `.md`, `.pdf`.

---

## 1. How to run

### Streamlit UI
```bash
cd intermediate/doc-qa-assistant
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```
Upload documents in the sidebar → **Build / rebuild index** → ask a question.

### CLI (no browser)
```bash
python cli.py ingest sample_docs/python_basics.txt sample_docs/study_notes.md
python cli.py ask --docs sample_docs/python_basics.txt --question "What is a list comprehension?" --show-sources
python cli.py ask --docs sample_docs/sample.pdf --question "What does a Zorblax reading above 40 indicate?"
```

### Verify offline (no key, no network)
```bash
python selfcheck.py     # 11 checks
pip install pytest && pytest -q
```

---

## 2. Pipeline and design decisions

### 2.1 Ingestion (`ingest.py`)
* TXT/Markdown read with encoding fallback (`utf-8` → `utf-8-sig` → `latin-1`).
* PDF text via `pypdf`; image-only PDFs raise a clear "no extractable text" error.
* Rejects unsupported extensions, oversized files (`MAX_FILE_MB`), empty files and missing paths.

### 2.2 Chunking (`chunking.py`)
**Word-window chunking with overlap** — default **140 words** per chunk, **30 words** overlap.
Rationale:
* word budget keeps each embedding focused and cost predictable;
* the 30-word overlap ensures a fact split across a boundary is still fully present in at least one chunk;
* chunks keep exact character offsets back into the source, so citations point at a real region.

### 2.3 Embeddings (`embeddings.py`)
* **Default: `hashing`** — deterministic feature hashing of content words with sub-linear term weighting and L2 normalisation. Dependency-free, offline, reproducible; ideal for lexical/semantic-ish retrieval on study notes.
* **Optional: `sentence-transformers`** (`all-MiniLM-L6-v2`) for true semantic embeddings.
* **Optional: `openai`** hosted embeddings.
Provider selection is automatic via `EMBEDDING_PROVIDER`, with safe fallback to hashing.

### 2.4 Vector store (`vector_store.py`)
* Default: dependency-free cosine-similarity list store (fine for hundreds of chunks).
* Optional: **FAISS** `IndexFlatIP` over L2-normalised vectors, used automatically when `faiss` is installed.

### 2.5 Grounded generation (`qa.py`)
* Retrieve top-k chunks, then answer **only** from them.
* With a real LLM: the system prompt forbids outside knowledge and requires `[n]` citations; the user prompt carries the numbered passages.
* Offline (`mock`): a deterministic **extractive** answerer copies the highest-overlap sentences from the retrieved chunks — grounded by construction.
* Every answer includes: the answer text, the **source chunks with scores and character offsets**, a **groundedness score** (share of answer terms present in the retrieved context), and a `refused` flag.

### 2.6 Anti-hallucination
1. Retrieval happens before generation; the model never answers from memory.
2. If the best similarity is below `MIN_SCORE` (default 0.05) **or the best chunk shares no content word with the question** (a lexical-anchoring check, `MIN_LEXICAL_OVERLAP`, default 1), the assistant **refuses**: *"The uploaded documents do not contain enough information to answer this."* The lexical check keeps the guardrail honest even with the hash-collision noise of the offline embedder (set it to 0 when using semantic embeddings).
3. Answers carry citations and a groundedness indicator; with the extractive engine groundedness is ~100%.

### 2.7 Error handling
Obvious failure modes become typed, friendly errors instead of crashes:
invalid file type / missing / empty / oversized PDF, empty question, unreachable LLM
(network), bad credentials, rate limit and server errors — the last three fall back
to the deterministic extractive answer with a visible warning, so the user still gets a grounded response.

---

## 3. Project layout

```
doc-qa-assistant/
├── app.py             # Streamlit UI
├── cli.py             # ingest / ask commands
├── qa.py              # retrieval + grounded generation
├── chunking.py        # word-window chunker with overlap
├── embeddings.py      # hashing / sentence-transformers / openai embedders
├── vector_store.py    # cosine list store (+ optional FAISS)
├── ingest.py          # TXT / MD / PDF ingestion
├── validators.py      # question validation
├── text_utils.py      # tokenisation helpers
├── config.py          # environment-driven settings
├── selfcheck.py       # offline self-check (11 checks)
├── tools/make_sample_pdf.py   # generates a valid demo PDF (stdlib only)
├── sample_docs/       # python_basics.txt, study_notes.md, sample.pdf
├── tests/             # pytest suite
├── requirements.txt
└── .env.example
```

---

## 4. Verified behaviour

| Command (no API key, no network) | Expected |
|---|---|
| `python selfcheck.py` | `11/11 checks passed`, exit `0` |
| `pytest -q` | all tests pass |
| `python cli.py ask --docs sample_docs/python_basics.txt --question "What is a list comprehension?"` | grounded answer + sources |
| `python cli.py ask ... --question "Who won the 1998 world cup?"` | `[REFUSED - nothing relevant found]` |

The self-check proves the key claim of this tier: retrieval finds the right chunk,
the answer comes from the document, and irrelevant questions are refused.

---

## 5. Live mode (OpenRouter)

Grounded answers are generated by **OpenRouter** as soon as a key is present:

```ini
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1   # or your own compatible gateway
LLM_MODEL=openai/gpt-4o-mini
```

The retrieved passages are injected into the prompt with numbered citations, and the
model is instructed to answer only from them. If the live call fails, the assistant
falls back to the deterministic extractive answer (still grounded) and shows a warning.
`.env` is gitignored.
