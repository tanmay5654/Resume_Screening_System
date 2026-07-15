"""
vector_store.py
----------------
Wraps ChromaDB -- the vector database that stores resume embeddings
and lets us search for the most similar ones to a job description.

This is the "vector search / RAG" piece the job description asks for
as a nice-to-have.
"""

import chromadb
from app.embeddings import embed_text, embed_batch

# Persistent local database -- data survives between runs, stored on disk.
_client = chromadb.PersistentClient(path="./chroma_data")

# IMPORTANT: explicitly use cosine distance, since our embeddings are
# normalized (normalize_embeddings=True in embeddings.py). Without this,
# ChromaDB defaults to squared L2 distance, which produces meaningless
# negative "similarity scores" once converted with (1 - distance) * 100.
_collection = _client.get_or_create_collection(
    name="resumes",
    metadata={"hnsw:space": "cosine"}
)


def clear_collection():
    """Wipe all stored resumes -- useful when starting a fresh screening batch."""
    global _collection
    _client.delete_collection(name="resumes")
    _collection = _client.get_or_create_collection(
        name="resumes",
        metadata={"hnsw:space": "cosine"}
    )


def add_resume(resume_id: str, filename: str, text: str):
    """
    Embed a resume's text and store it in the vector database,
    along with metadata (filename) so we can identify it later.
    """
    vector = embed_text(text)
    _collection.add(
        ids=[resume_id],
        embeddings=[vector],
        documents=[text],
        metadatas=[{"filename": filename}],
    )


def add_resumes_batch(resumes: list[dict]):
    """
    resumes: list of {"id": ..., "filename": ..., "text": ...}
    Batch version -- faster than adding one at a time.
    """
    texts = [r["text"] for r in resumes]
    vectors = embed_batch(texts)
    _collection.add(
        ids=[r["id"] for r in resumes],
        embeddings=vectors,
        documents=texts,
        metadatas=[{"filename": r["filename"]} for r in resumes],
    )


def list_all_resumes() -> list[dict]:
    """
    Return every resume currently stored in the vector database.
    Useful for debugging -- e.g. checking that an upload actually worked.
    """
    all_items = _collection.get()
    ids = all_items["ids"]
    metadatas = all_items["metadatas"]
    documents = all_items["documents"]

    resumes = []
    for i in range(len(ids)):
        resumes.append({
            "id": ids[i],
            "filename": metadatas[i]["filename"],
            "text_preview": documents[i][:200],  # first 200 chars only
        })
    return resumes


def find_best_matches(job_description: str, top_k: int = 10) -> list[dict]:
    """
    Embed the job description, then find the top_k most similar
    resumes stored in the vector database.

    Returns a list of dicts: {id, filename, text, similarity_score}
    """
    jd_vector = embed_text(job_description)

    results = _collection.query(
        query_embeddings=[jd_vector],
        n_results=top_k,
    )

    matches = []
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    # With cosine space, ChromaDB's "distance" is (1 - cosine_similarity),
    # so it naturally ranges from 0 (identical) to 2 (opposite).
    distances = results.get("distances", [[None] * len(ids)])[0]

    for i in range(len(ids)):
        distance = distances[i]
        # Convert distance to a friendlier 0-100 similarity score
        similarity_score = None
        if distance is not None:
            similarity_score = round((1 - distance) * 100, 1)

        matches.append({
            "id": ids[i],
            "filename": metadatas[i]["filename"],
            "text": documents[i],
            "similarity_score": similarity_score,
        })

    return matches