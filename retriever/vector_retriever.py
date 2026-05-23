"""
vector_retriever.py — FAISS vector retriever + BM25 hybrid retrieval

Patterns:
  - Dense retrieval (embedding similarity via FAISS)
  - Sparse retrieval (BM25 keyword matching)
  - Hybrid retrieval (RRF fusion of dense + sparse)
  - Reciprocal Rank Fusion (RRF) for result merging
"""
import math
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from openai import OpenAI

# FAISS optional — falls back to numpy dot product if not installed
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

client = OpenAI()
EMBED_MODEL = "text-embedding-3-small"


@dataclass
class SearchResult:
    chunk_id: int
    text: str
    score: float
    metadata: dict = field(default_factory=dict)
    source: str = "unknown"  # "dense", "sparse", or "hybrid"


# ── Embedding helper ──────────────────────────────────────────────────────────

def embed_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Embed texts in batches to stay within API limits."""
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_vectors.extend([item.embedding for item in response.data])
    return all_vectors


# ── Dense Retriever (FAISS) ───────────────────────────────────────────────────

class DenseRetriever:
    """FAISS-based dense retriever using OpenAI embeddings."""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self._index = None
        self._texts: list[str] = []
        self._metadata: list[dict] = []

    def add(self, texts: list[str], metadata: list[dict] | None = None) -> None:
        vectors = embed_batch(texts)
        arr = np.array(vectors, dtype=np.float32)

        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.where(norms == 0, 1, norms)

        if self._index is None:
            if FAISS_AVAILABLE:
                self._index = faiss.IndexFlatIP(self.dimension)
            else:
                self._index = arr  # numpy fallback
        elif FAISS_AVAILABLE:
            self._index.add(arr)
        else:
            self._index = np.vstack([self._index, arr])

        if FAISS_AVAILABLE and self._index.ntotal == len(arr):
            self._index.add(arr)

        self._texts.extend(texts)
        self._metadata.extend(metadata or [{} for _ in texts])

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self._texts:
            return []

        [q_vec] = embed_batch([query])
        q_arr = np.array([q_vec], dtype=np.float32)
        q_arr = q_arr / np.linalg.norm(q_arr)

        if FAISS_AVAILABLE and isinstance(self._index, faiss.Index):
            scores, indices = self._index.search(q_arr, min(top_k, len(self._texts)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    results.append(SearchResult(
                        chunk_id=idx,
                        text=self._texts[idx],
                        score=float(score),
                        metadata=self._metadata[idx],
                        source="dense",
                    ))
        else:
            # Numpy fallback
            scores = (self._index @ q_arr.T).squeeze()
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = [
                SearchResult(
                    chunk_id=int(i),
                    text=self._texts[i],
                    score=float(scores[i]),
                    metadata=self._metadata[i],
                    source="dense",
                )
                for i in top_indices
            ]

        return results

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        if FAISS_AVAILABLE and isinstance(self._index, faiss.Index):
            faiss.write_index(self._index, os.path.join(path, "index.faiss"))
        else:
            np.save(os.path.join(path, "index.npy"), self._index)
        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump({"texts": self._texts, "metadata": self._metadata}, f)

    @classmethod
    def load(cls, path: str) -> "DenseRetriever":
        r = cls()
        if FAISS_AVAILABLE:
            r._index = faiss.read_index(os.path.join(path, "index.faiss"))
        else:
            r._index = np.load(os.path.join(path, "index.npy"))
        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        r._texts = meta["texts"]
        r._metadata = meta["metadata"]
        return r


# ── Sparse Retriever (BM25) ───────────────────────────────────────────────────

class BM25Retriever:
    """BM25 keyword retriever — no API calls, pure Python."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._texts: list[str] = []
        self._metadata: list[dict] = []
        self._tf: list[dict[str, float]] = []       # term frequency per doc
        self._df: dict[str, int] = defaultdict(int)  # document frequency
        self._avgdl: float = 0.0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower()) if text else []

    def add(self, texts: list[str], metadata: list[dict] | None = None) -> None:
        import re  # local import to keep module clean
        self._texts.extend(texts)
        self._metadata.extend(metadata or [{} for _ in texts])

        for text in texts:
            tokens = self._tokenize(text)
            tf: dict[str, float] = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            for token in tf:
                self._df[token] += 1
            self._tf.append(dict(tf))

        total_len = sum(sum(tf.values()) for tf in self._tf)
        self._avgdl = total_len / len(self._tf) if self._tf else 1.0

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        import re
        if not self._texts:
            return []

        query_tokens = self._tokenize(query)
        N = len(self._texts)
        scores = []

        for i, tf in enumerate(self._tf):
            doc_len = sum(tf.values())
            score = 0.0
            for token in query_tokens:
                if token not in tf:
                    continue
                df = self._df.get(token, 0)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                tf_score = tf[token] * (self.k1 + 1) / (
                    tf[token] + self.k1 * (1 - self.b + self.b * doc_len / self._avgdl)
                )
                score += idf * tf_score
            scores.append(score)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                chunk_id=i,
                text=self._texts[i],
                score=scores[i],
                metadata=self._metadata[i],
                source="sparse",
            )
            for i in top_indices
            if scores[i] > 0
        ]


# ── Hybrid Retriever (RRF fusion) ─────────────────────────────────────────────

class HybridRetriever:
    """Combine dense + sparse results using Reciprocal Rank Fusion (RRF).

    RRF score = sum(1 / (k + rank_i)) for each ranklist i
    Better than score normalization — rank-based, no calibration needed.
    """

    def __init__(self, rrf_k: int = 60, dense_weight: float = 0.6):
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.dense = DenseRetriever()
        self.sparse = BM25Retriever()

    def add(self, texts: list[str], metadata: list[dict] | None = None) -> None:
        self.dense.add(texts, metadata)
        self.sparse.add(texts, metadata)

    def search(self, query: str, top_k: int = 5, fetch_k: int = 20) -> list[SearchResult]:
        dense_results = self.dense.search(query, top_k=fetch_k)
        sparse_results = self.sparse.search(query, top_k=fetch_k)

        # RRF fusion
        rrf_scores: dict[int, float] = defaultdict(float)
        result_map: dict[int, SearchResult] = {}

        for rank, r in enumerate(dense_results):
            rrf_scores[r.chunk_id] += self.dense_weight / (self.rrf_k + rank + 1)
            result_map[r.chunk_id] = r

        for rank, r in enumerate(sparse_results):
            rrf_scores[r.chunk_id] += (1 - self.dense_weight) / (self.rrf_k + rank + 1)
            if r.chunk_id not in result_map:
                result_map[r.chunk_id] = r

        top = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        fused = []
        for chunk_id, score in top:
            r = result_map[chunk_id]
            fused.append(SearchResult(
                chunk_id=chunk_id,
                text=r.text,
                score=score,
                metadata=r.metadata,
                source="hybrid",
            ))
        return fused
