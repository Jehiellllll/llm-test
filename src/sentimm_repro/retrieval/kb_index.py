from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class RetrievalResult:
    indices: np.ndarray
    distances: np.ndarray
    label_distribution: np.ndarray


class KBIndex:
    def __init__(self, top_k: int = 5, backend: str = "faiss"):
        self.top_k = top_k
        self.backend = backend
        self.labels = None
        self._faiss = None
        self._nn = None
        self._use_faiss = False

    def fit(self, vectors: np.ndarray, labels: np.ndarray):
        self.labels = labels
        if self.backend == "faiss":
            try:
                import faiss

                self._faiss = faiss.IndexFlatL2(vectors.shape[1])
                self._faiss.add(vectors.astype(np.float32))
                self._use_faiss = True
            except Exception:
                self._use_faiss = False

        if not self._use_faiss:
            self._nn = NearestNeighbors(n_neighbors=min(self.top_k, len(vectors)), metric="euclidean")
            self._nn.fit(vectors)

    def query(self, vectors: np.ndarray, num_classes: int) -> RetrievalResult:
        if self._use_faiss:
            distances, indices = self._faiss.search(vectors.astype(np.float32), self.top_k)
        else:
            distances, indices = self._nn.kneighbors(vectors, n_neighbors=self.top_k, return_distance=True)

        dists = np.zeros((vectors.shape[0], num_classes), dtype=np.float32)
        for i in range(indices.shape[0]):
            neigh = indices[i]
            weights = 1.0 / (distances[i] + 1e-6)
            for j, idx in enumerate(neigh):
                label = self.labels[idx]
                dists[i, label] += weights[j]
            if dists[i].sum() > 0:
                dists[i] /= dists[i].sum()
        return RetrievalResult(indices=indices, distances=distances, label_distribution=dists)
