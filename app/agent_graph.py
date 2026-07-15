"""
agent_graph.py
--------------
Orchestrates the resume-screening pipeline as a LangGraph state machine,
mirroring the same pattern as your Research & Report Agent:
    search/retrieve -> analyze -> write

Here it's:
    retrieve (vector search) -> explain (LLM grounds its verdict per resume) -> rank

Keeping the same "agentic, stateful, multi-step" structure gives you a
consistent story to tell across both projects.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END

from app.vector_store import find_best_matches
from app.llm_explain import explain_match


class ScreeningState(TypedDict):
    job_description: str
    top_k: int
    candidates: list        # raw vector-search results
    results: list            # candidates + LLM explanation, ranked


def retrieve_node(state: ScreeningState) -> ScreeningState:
    """Step 1: vector search for the resumes most similar to the JD."""
    matches = find_best_matches(state["job_description"], top_k=state["top_k"])
    state["candidates"] = matches
    return state


def explain_node(state: ScreeningState) -> ScreeningState:
    """Step 2: for each retrieved candidate, ask the LLM to ground a verdict."""
    results = []
    for candidate in state["candidates"]:
        explanation = explain_match(state["job_description"], candidate["text"])
        results.append({
            "filename": candidate["filename"],
            "similarity_score": candidate["similarity_score"],
            "verdict": explanation["verdict"],
            "reasoning": explanation["reasoning"],
            "missing": explanation["missing"],
        })
    state["results"] = results
    return state


def rank_node(state: ScreeningState) -> ScreeningState:
    """Step 3: order results -- strongest matches first."""
    verdict_order = {"Strong Match": 0, "Good Match": 1, "Weak Match": 2, "Not a Match": 3}
    state["results"].sort(
        key=lambda r: (verdict_order.get(r["verdict"], 4), -(r["similarity_score"] or 0))
    )
    return state


def build_graph():
    graph = StateGraph(ScreeningState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("explain", explain_node)
    graph.add_node("rank", rank_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "explain")
    graph.add_edge("explain", "rank")
    graph.add_edge("rank", END)

    return graph.compile()


_compiled_graph = build_graph()


def run_screening(job_description: str, top_k: int = 10) -> list[dict]:
    """Entry point used by the API layer."""
    initial_state: ScreeningState = {
        "job_description": job_description,
        "top_k": top_k,
        "candidates": [],
        "results": [],
    }
    final_state = _compiled_graph.invoke(initial_state)
    return final_state["results"]
