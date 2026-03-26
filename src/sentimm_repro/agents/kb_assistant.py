from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from sentimm_repro.retrieval.kb_index import KBIndex, RetrievalResult


class KBAssistant:
    def __init__(self, top_k: int = 5, backend: str = "faiss"):
        self.vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), strip_accents="unicode")
        self.index = KBIndex(top_k=top_k, backend=backend)

    def build(self, kb_texts: list[str], labels: np.ndarray):
        kb_emb = self.vectorizer.fit_transform(kb_texts).toarray().astype(np.float32)
        self.index.fit(kb_emb, labels)
        return kb_emb

    def query(self, kb_texts: list[str], num_classes: int) -> tuple[np.ndarray, RetrievalResult]:
        q = self.vectorizer.transform(kb_texts).toarray().astype(np.float32)
        result = self.index.query(q, num_classes=num_classes)
        return q, result
