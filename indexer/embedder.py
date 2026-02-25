"""
Code embedding using sentence-transformers.

The model is loaded lazily on first use (avoids slow import at startup).
We use ``all-MiniLM-L6-v2`` (22 MB) by default — lightweight and fast.
"""

from __future__ import annotations

from typing import List

import numpy as np

from config import EMBED_MODEL

_model = None  # module-level singleton


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of strings.
    Returns a float32 ndarray of shape (len(texts), embedding_dim).
    Embeddings are L2-normalised for cosine similarity via inner product.
    """
    model = _get_model()
    return np.array(
        model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )


def embed_single(text: str) -> np.ndarray:
    """Convenience wrapper — embed a single string."""
    return embed_texts([text])[0]
