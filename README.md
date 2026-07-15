# AI Resume Screening System

An AI agent that screens resumes against a job description, using retrieval-augmented generation (RAG) built on real embeddings and a vector database — grounded, ranked, and explained in plain language.

Given a batch of resumes and a job description, the system retrieves the most semantically relevant resumes via vector search, then uses an LLM to generate a grounded verdict for each candidate: a match rating, a plain-language explanation, and a list of gaps — all strictly based on what's actually written in the resume, not inferred or invented.

**Live demo:** _add your deployed URL here_
**Backend repo / this repo:** _add your GitHub URL here_

---

## Why this project

This was built to close a specific gap: a companion project of mine, a [Research & Report Agent](#), uses live web search (Tavily) for retrieval rather than embeddings or a vector database. This project fills that gap with a genuinely useful, real-world use case — resume screening — while reusing the same agentic, multi-step architecture (retrieve → reason → generate) as the original.

Together, the two projects cover both major flavours of RAG:
- **Search-based RAG** (Research & Report Agent) — retrieves live web content
- **Embedding-based RAG** (this project) — retrieves from a stored, embedded knowledge base

## How it works

```
Resumes (PDF / DOCX / TXT)
        │
        ▼
   Parse to text            (pypdf, python-docx)
        │
        ▼
   Embed                    (sentence-transformers, local, free)
        │
        ▼
   Store in vector DB       (ChromaDB, cosine similarity)


Job description
        │
        ▼
   Embed (same model)
        │
        ▼
   Vector search ───────────► top-k most similar resumes
        │
        ▼
   LLM grounds a verdict per resume   (Llama 3.3 70B via Groq)
        │
        ▼
   Rank results (Strong → Good → Weak → Not a Match)
        │
        ▼
   Ranked, explained results returned to the frontend
```

The retrieval → reasoning pipeline is orchestrated as a **LangGraph** state machine with three nodes — `retrieve`, `explain`, `rank` — mirroring the same agentic structure used in my other project.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Parsing | pypdf, python-docx | Extract plain text from uploaded resumes |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Free, local, no API key required |
| Vector database | ChromaDB (cosine similarity) | Zero-setup, file-based, ideal for a focused project |
| Orchestration | LangGraph | Stateful, multi-step agent workflow |
| LLM | Llama 3.3 70B via Groq | Fast, free inference; low temperature for consistent judging |
| Backend | FastAPI | Async REST API |
| Frontend | HTML / CSS / vanilla JS | Lightweight, no build step, deploys anywhere |

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/yourusername/resume-screener.git
cd resume-screener
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure your API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your-groq-api-key
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Run the backend

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Open the frontend

Open `frontend/index.html` directly in your browser. Update the `API_BASE` constant near the top of its `<script>` tag if your backend isn't running on `localhost:8000`.

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload-resumes` | Upload one or more resumes; parses, embeds, and stores them |
| `POST` | `/screen` | Given a job description, returns ranked, explained candidate matches |
| `GET` | `/resumes` | List every resume currently stored (debugging) |
| `POST` | `/reset` | Clear all stored resumes |
| `GET` | `/health` | Health check |

## Design decisions worth knowing

- **Grounding by design** — the LLM's system prompt explicitly restricts it to facts present in the resume text, and instructs it to flag missing information rather than guess. This mirrors a hallucination problem I hit and fixed in my other project, applied here from day one instead of patched in afterward.
- **Cosine similarity, explicitly configured** — ChromaDB defaults to squared L2 (Euclidean) distance, which is incompatible with the normalized embeddings this project uses and produces meaningless (even negative) similarity scores. The collection is explicitly configured with `metadata={"hnsw:space": "cosine"}` to fix this.
- **Low LLM temperature (0.2)** — resume screening should be consistent and factual, not creative.

## Known limitations & next steps

- No automated evaluation harness yet — currently verified by manual review. Next step: a test set of resumes with known-correct verdicts, checked automatically.
- Each retrieved resume triggers its own LLM call — fine at small scale; would benefit from caching and/or batching for larger candidate pools.
- ChromaDB's local, file-based storage is ideal for a single-instance project; a hosted vector database (Pinecone) or `pgvector` would be the natural next step for multi-server deployment.
- Frontend is intentionally dependency-free (no build step); a React port is a natural next iteration if this grows.

## License

MIT