"""
main.py
-------
FastAPI backend for the AI Resume Screening System.

Endpoints:
  POST /upload-resumes   -- upload one or more resumes (PDF/DOCX/TXT),
                              parses + embeds + stores them
  POST /screen            -- given a job description, retrieves the best
                              matching resumes and asks the LLM to explain
                              each verdict
  POST /reset              -- clear all stored resumes (start a fresh batch)
  GET  /resumes            -- debug: list every resume currently stored
  GET  /health             -- simple health check
"""

from dotenv import load_dotenv
load_dotenv()  # reads GROQ_API_KEY (and anything else) from a local .env file

import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.parser import extract_text
from app.vector_store import add_resumes_batch, clear_collection, list_all_resumes
from app.agent_graph import run_screening

app = FastAPI(title="AI Resume Screening System")

# Allow the React frontend (running on a different port/domain) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your real frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScreenRequest(BaseModel):
    job_description: str
    top_k: int = 10


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset():
    clear_collection()
    return {"status": "cleared"}


@app.get("/resumes")
def resumes():
    """Debug endpoint: list every resume currently stored in the vector database."""
    return {"resumes": list_all_resumes()}


@app.post("/upload-resumes")
async def upload_resumes(files: list[UploadFile] = File(...)):
    """
    Accepts multiple resume files, extracts text, embeds them,
    and stores them in the vector database.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    parsed_resumes = []
    failed = []

    for file in files:
        file_bytes = await file.read()
        try:
            text = extract_text(file.filename, file_bytes)
            if not text.strip():
                failed.append({"filename": file.filename, "reason": "No extractable text"})
                continue
            parsed_resumes.append({
                "id": str(uuid.uuid4()),
                "filename": file.filename,
                "text": text,
            })
        except ValueError as e:
            failed.append({"filename": file.filename, "reason": str(e)})

    if parsed_resumes:
        add_resumes_batch(parsed_resumes)

    return {
        "uploaded": len(parsed_resumes),
        "failed": failed,
    }


@app.post("/screen")
def screen(request: ScreenRequest):
    """
    Given a job description, runs the full agent:
    retrieve top matching resumes -> LLM explains each verdict -> ranked results.
    """
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="job_description is required")

    results = run_screening(request.job_description, top_k=request.top_k)
    return {"results": results}