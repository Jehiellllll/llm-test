from __future__ import annotations

import numpy as np

from sentimm_repro.models.torch_head import TorchHeadWrapper


class ClassifierAggregator:
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, mode: str = "weighted"):
        self.alpha = alpha
        self.beta = beta
        self.mode = mode
        self.learned_head: TorchHeadWrapper | None = None

    def fit(self, multimodal_score: np.ndarray, retrieved_score: np.ndarray, labels: np.ndarray, head_kwargs: dict | None = None):
        if self.mode == "learned":
            x = np.concatenate([multimodal_score, retrieved_score], axis=1)
            self.learned_head = TorchHeadWrapper(in_dim=x.shape[1], out_dim=multimodal_score.shape[1], **(head_kwargs or {}))
            self.learned_head.fit(x, labels)

    def combine(self, multimodal_score: np.ndarray, retrieved_score: np.ndarray) -> np.ndarray:
        if self.mode == "learned" and self.learned_head is not None:
            x = np.concatenate([multimodal_score, retrieved_score], axis=1)
            return self.learned_head.predict_proba(x)

        z = self.alpha * multimodal_score + self.beta * retrieved_score
        z = z / (z.sum(axis=1, keepdims=True) + 1e-8)
        return z
