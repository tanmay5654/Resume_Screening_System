"""
llm_explain.py
--------------
Uses an LLM (Llama 3.3 70B via Groq -- same choice as your other project,
for a consistent story) to read a resume alongside the job description
and produce a grounded, human-readable explanation of the match.

This is the "generation grounded in retrieved evidence" step -- same
RAG principle as your web-search agent, just with resumes as the
retrieved evidence instead of web search results.
"""

import os
from groq import Groq

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key at https://console.groq.com and set it "
                "before running the server."
            )
        _client = Groq(api_key=api_key)
    return _client


SYSTEM_PROMPT = """You are a careful, fair technical recruiter assistant.
You will be given a job description and a single candidate's resume text.

Your job:
1. Judge how well this specific resume matches the job description.
2. Only use facts that are actually present in the resume text below --
   never invent skills, years of experience, or qualifications that
   are not explicitly stated or clearly implied by the resume.
3. If the resume is missing information needed to judge a requirement,
   say so explicitly rather than assuming the candidate has it.

Respond in this exact format:
VERDICT: <Strong Match | Good Match | Weak Match | Not a Match>
REASONING: <2-3 sentences, grounded only in what's actually in the resume text>
MISSING: <key requirements from the job description that this resume does not
clearly show, or "None obvious" if it covers everything>
"""


def explain_match(job_description: str, resume_text: str) -> dict:
    """
    Ask the LLM to explain, in grounded terms, how well a resume matches
    a job description. Returns a structured dict.
    """
    client = get_client()

    user_prompt = f"""JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME:
{resume_text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # low temperature: we want consistent, factual judging
        max_tokens=400,
    )

    raw_text = response.choices[0].message.content

    # Simple parsing of the structured response
    verdict, reasoning, missing = "Unknown", raw_text, "Unknown"
    for line in raw_text.splitlines():
        if line.upper().startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip()
        elif line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
        elif line.upper().startswith("MISSING:"):
            missing = line.split(":", 1)[1].strip()

    return {
        "verdict": verdict,
        "reasoning": reasoning,
        "missing": missing,
        "raw": raw_text,
    }
