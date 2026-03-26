from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from sentimm_repro.models.torch_head import TorchHeadWrapper
from sentimm_repro.types import AnalystOutput


class TextAnalyst:
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), strip_accents="unicode")
        self.head: TorchHeadWrapper | None = None

    def fit(self, texts: list[str], labels: np.ndarray, num_classes: int, head_kwargs: dict):
        x = self.vectorizer.fit_transform(texts).toarray()
        self.head = TorchHeadWrapper(in_dim=x.shape[1], out_dim=num_classes, **head_kwargs)
        self.head.fit(x, labels)
        return self.forward(texts)

    def forward(self, texts: list[str]) -> AnalystOutput:
        x = self.vectorizer.transform(texts).toarray()
        score = self.head.predict_proba(x) if self.head else np.zeros((len(texts), 7), dtype=np.float32)
        rationale = [f"text_len={len(t)}" for t in texts]
        return AnalystOutput(embedding=x, score=score, rationale=rationale)
