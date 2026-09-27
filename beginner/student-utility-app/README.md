# StudyMate — AI-Powered Student Utility App (Beginner Tier)

StudyMate is a small but complete AI-powered study tool built around an LLM API.
It is **not** a generic chatbot: every action runs a purpose-built prompt for a
specific student workflow and returns a predictable, sectioned result.

Supported utility features:

| Feature | What it does |
|---|---|
| **Summarise notes** | Extractive-style summary → key points + a one-line takeaway |
| **Generate a quiz** | Active-recall questions + answer key from your notes |
| **Improve a weak answer** | Rewrites a draft answer into a stronger one, and explains what changed |
| **Explain a concept** | Explains a concept at three depths + a common mistake to avoid |

---

## 1. How to run

### Option A — Streamlit UI (recommended)
```bash
cd beginner/student-utility-app
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```
Then open the URL Streamlit prints (default http://localhost:8501).

### Option B — CLI (no browser needed)
```bash
python cli.py --list-features
python cli.py --feature summarize --input-file notes.txt
python cli.py --feature quiz --input "Mitosis is the process of cell division..." --num-questions 4
python cli.py --feature improve --input "I think photosynthesis is kind of important..." --topic photosynthesis
python cli.py --feature explain --input-file notes.txt --topic "Calvin cycle"
```

### Option C — verify it offline (no API key, no network)
```bash
python selfcheck.py         # 10 checks, exits 0 on success
pip install pytest && pytest -q
```

---

## 2. Providers and offline mode

Configuration is environment driven (see `.env.example`).

| `LLM_PROVIDER` | Requires | Behaviour |
|---|---|---|
| `openrouter` | `OPENROUTER_API_KEY` (+ optional `OPENROUTER_BASE_URL`) | **OpenRouter** (recommended for live use) or any OpenRouter-compatible gateway |
| `openai` | `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`) | Any OpenAI-compatible `/chat/completions` endpoint |
| `zai` | `ZAI_API_KEY` (+ optional `ZAI_BASE_URL`) | Zhipu GLM (OpenAI-compatible) |
| `mock` | nothing | **Offline deterministic engine** — no key, no network |

If `LLM_PROVIDER` is unset, the app picks `openai` when a key is present,
otherwise `zai`, otherwise `mock`. **With no configuration at all the app still
works** and every feature returns real output via the offline engine.

The offline engine (`offline_engine.py`) is not a stub. It implements genuine
deterministic NLP heuristics:

* **summary** — frequency-scored extractive summarisation (sentence scoring over content words, lead-sentence boost, original order preserved);
* **quiz** — definition-sentence extraction ("X is…") + cloze deletion of key terms with distractor options;
* **improve** — rule-based rewriting: hedge removal, precision, restructuring into claim → support → summary, plus a changelog;
* **explain** — definition spotting + key-term scaffolding, grounded only in the supplied material.

---

## 3. Prompt design

Prompts live in `prompts.py` and follow four rules:

1. **Role priming** — "You are StudyMate, a focused study assistant…".
2. **Explicit output contract** — each feature states the exact Markdown sections it must return, so the UI renders predictably.
3. **Grounding guard** — the model must stay within the supplied material and say *"not covered in the provided material"* rather than invent facts.
4. **Audience/tone control** — concise, exam-oriented, no filler.

Example (quiz system prompt):
```text
You are StudyMate, ... Work only from the material the student provides ...
Return exactly this structure:
## Questions
1. <question>  (mix multiple-choice and short answer; 4 to 6 questions)
## Answer key
1. <answer with a one-line justification>
```

---

## 4. Validation and error handling

* `validators.py` rejects empty / whitespace-only / non-text input, input below
  `MIN_INPUT_CHARS` (default 15) and above `MAX_INPUT_CHARS` (default 12000).
  The UI shows the exact reason; the provider is never called for invalid input.
* `llm_client.py` converts provider failures into a typed `LLMError` with a safe
  message: missing key (`auth`), unreachable service (`network`), rate limit
  (`rate_limit`, with retry backoff), server error (`server`), unexpected status
  (`http`), unreadable response (`parse`), empty answer (`empty`).
* `core.run_feature()` never raises — it returns `{"ok": False, "error": ..., "error_kind": ...}`
  so the UI can render the message and keep working.

---

## 5. Project layout

```
student-utility-app/
├── app.py              # Streamlit UI
├── cli.py              # Command-line runner
├── core.py             # validate -> prompt -> generate -> format orchestration
├── prompts.py          # per-feature prompt templates + output contracts
├── validators.py       # input validation
├── llm_client.py       # provider abstraction (openai / zai / mock)
├── offline_engine.py   # deterministic offline generation
├── config.py           # environment-driven settings
├── selfcheck.py        # offline self-check (10 checks)
├── tests/              # pytest suite
├── requirements.txt
├── .env.example
└── README.md
```

---

## 6. Verified behaviour

Run from this folder with no API key present:

| Command | Expected result |
|---|---|
| `python selfcheck.py` | `10/10 checks passed`, exit code `0` |
| `pytest -q` | all tests pass |
| `python cli.py --feature summarize --input "..."` | prints `## Key points` + `## One-line takeaway` |
| `python -c "import app"` | imports cleanly |

---

## 7. Dependencies

`streamlit`, `requests`, `python-dotenv` (optional), `pytest` (tests only).
No ML frameworks, no vendor SDKs, no network required for offline mode.

---

## 8. Live mode (OpenRouter)

The app is wired for **OpenRouter** out of the box. Put your key in `.env`:

```ini
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1   # or your own compatible gateway
LLM_MODEL=openai/gpt-4o-mini                        # any OpenRouter model id
```

As soon as the key is present the same features run against the live model; if the
call fails the UI shows a friendly error and the app keeps working. `.env` is
gitignored - `dotenv` is optional, so the app also runs with plain environment variables.
