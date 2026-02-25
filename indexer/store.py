"""
FAISS-backed vector store for code chunks.

Stores embeddings + JSON metadata side-by-side.
Supports save/load for persistent sessions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from config import METADATA_PATH, VECTOR_DB_PATH


class VectorStore:
    """
    Wraps a FAISS IndexFlatIP (inner-product = cosine on normalised vecs).

    Usage::

        store = VectorStore()
        store.add(embeddings_array, metadata_list)
        results = store.search(query_embedding, k=5)
        store.save()

        store2 = VectorStore()
        store2.load()
    """

    def __init__(self, dim: int = 384):
        import faiss  # lazy import so the rest of the app works w/o faiss
        self.dim   = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict[str, Any]] = []
        VECTOR_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Write ────────────────────────────────────────────────────────────────

    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add *embeddings* (shape N × dim) and corresponding *metadata* dicts."""
        if embeddings.shape[0] != len(metadata):
            raise ValueError("embeddings and metadata must have the same length")
        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadata)

    # ── Read ─────────────────────────────────────────────────────────────────

    def search(self, query: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Return up to *k* nearest neighbours, each as a metadata dict with an
        added ``score`` key (higher = more similar).
        """
        if self.index.ntotal == 0:
            return []

        q = query.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(q, min(k, self.index.ntotal))

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                item = dict(self.metadata[idx])
                item["score"] = float(dist)
                results.append(item)
        return results

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        import faiss
        faiss.write_index(self.index, str(VECTOR_DB_PATH) + ".faiss")
        METADATA_PATH.write_text(json.dumps(self.metadata, indent=2))

    def load(self) -> None:
        import faiss
        idx_path = Path(str(VECTOR_DB_PATH) + ".faiss")
        if idx_path.exists():
            self.index = faiss.read_index(str(idx_path))
        if METADATA_PATH.exists():
            self.metadata = json.loads(METADATA_PATH.read_text())

    @property
    def size(self) -> int:
        return self.index.ntotal
