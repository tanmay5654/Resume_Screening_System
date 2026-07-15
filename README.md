# AI Resume Screening System

An AI agent that reads a job description and a batch of resumes, retrieves
the best-matching resumes using vector search, and asks an LLM to explain
*why* each one matches — grounded strictly in what's actually in the resume.

Built to fill the one real gap in my other project (the Research & Report
Agent): hands-on experience with embeddings and a vector database, since
that project used live web search (Tavily) instead.

## Architecture

```
Resumes (PDF/DOCX/TXT)
    |
    v
Parse -> extract plain text        (app/parser.py)
    |
    v
Embed -> sentence-transformers      (app/embeddings.py)
    |
    v
Store -> ChromaDB (vector database) (app/vector_store.py)

Job Description
    |
    v
LangGraph agent (app/agent_graph.py):
    retrieve  -> vector search for closest resumes
    explain   -> LLM (Llama 3.3 70B / Groq) grounds a verdict per resume,
                 using ONLY the resume text as evidence
    rank      -> order by verdict, then similarity score
    |
    v
FastAPI backend (app/main.py) -> JSON response
    |
    v
Frontend (frontend/index.html) -> displays ranked, explained results
```

This mirrors the same agentic, multi-step structure as my other project
(search -> analyze -> write), just applied to resume screening
(retrieve -> explain -> rank).

## Why these choices

- **sentence-transformers (local, free)** for embeddings — no API key or
  cost needed to get real embeddings/vector-search experience.
- **ChromaDB** — lightweight, runs locally, zero setup, perfect for a
  portfolio project (swap for Pinecone/pgvector later if needed).
- **Groq + Llama 3.3 70B** for the LLM step — same choice as my other
  project, fast and free.
- **Low temperature (0.2)** on the LLM call — for consistent, factual
  judging rather than creative variation.
- **Grounding by design** — the system prompt explicitly tells the model
  to only use facts present in the resume text, and to say when
  information is missing rather than guess. Same lesson from the
  hallucination problem in my other project, applied here from day one.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Get a free Groq API key at https://console.groq.com and set it:
   ```
   export GROQ_API_KEY="your-key-here"
   ```

3. Run the backend:
   ```
   uvicorn app.main:app --reload --port 8000
   ```

4. Open `frontend/index.html` directly in your browser (no build step
   needed — it's plain HTML/JS that calls the backend on localhost:8000).

5. Upload a few resumes, paste a job description, and click "Screen
   Candidates."

## What I'd add next (for the interview "what would you improve" question)

- An evaluation harness: a small set of resumes with known correct
  verdicts, checked automatically to catch regressions.
- Batch scaling: currently each resume gets its own LLM call for the
  explanation step — for large batches, I'd only send the LLM call for
  the top-N retrieved by vector search (which is already how it works),
  and add caching for repeated job descriptions.
- Swap ChromaDB for a hosted vector database (Pinecone) or pgvector if
  this needed to run across multiple servers rather than a single
  local instance.
- A proper React frontend (currently a lightweight static HTML page to
  keep the project simple and fast to build).

## Interview talking points

- **"Why did you build this?"** — To get real, hands-on experience with
  embeddings and vector databases, which my other project didn't need
  since it used live web search instead.
- **"How is this a RAG system?"** — Retrieval: vector search finds the
  most relevant resumes for a job description. Generation: the LLM
  explains the match, grounded only in the retrieved resume's actual
  text.
- **"How do you prevent hallucination here?"** — The system prompt
  explicitly restricts the LLM to facts in the resume text, and asks it
  to flag missing information rather than assume it. Same principle as
  my other project, applied from the start this time instead of added
  after the fact.
