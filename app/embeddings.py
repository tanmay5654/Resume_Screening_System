"""
embeddings.py
-------------
Turns text into embeddings (numeric vectors) using a free, local model
(sentence-transformers). No API key needed for this step -- it runs
entirely on your machine.

This is the piece your web-search-based agent didn't have: real,
hands-on embeddings + vector search experience.
"""

from sentence_transformers import SentenceTransformer

# Loaded once and reused -- loading the model is the slow part,
# so we don't want to reload it for every resume.
_model = None


def get_model():
    global _model
    if _model is None:
        # all-MiniLM-L6-v2: small, fast, good enough for semantic similarity.
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_text(text: str) -> list[float]:
    """Convert a single piece of text into an embedding vector."""
    model = get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Convert many pieces of text into embeddings in one batch (faster)."""
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()
