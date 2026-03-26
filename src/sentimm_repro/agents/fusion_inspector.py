from __future__ import annotations

import numpy as np

from sentimm_repro.models.torch_head import TorchHeadWrapper
from sentimm_repro.types import AnalystOutput


class FusionInspector:
    def __init__(self, discrepancy_threshold: float = 0.25, refinement_strength: float = 0.3):
        self.discrepancy_threshold = discrepancy_threshold
        self.refinement_strength = refinement_strength
        self.head: TorchHeadWrapper | None = None

    def fuse_embedding(self, text_emb: np.ndarray, image_emb: np.ndarray) -> np.ndarray:
        return np.concatenate([text_emb, image_emb], axis=1)

    def discrepancy(self, fusion_score: np.ndarray, text_score: np.ndarray, image_score: np.ndarray) -> np.ndarray:
        unimodal_mean = 0.5 * (text_score + image_score)
        return np.abs(fusion_score - unimodal_mean).mean(axis=1)

    def refine(self, fusion_score: np.ndarray, text_score: np.ndarray, image_score: np.ndarray) -> np.ndarray:
        d = self.discrepancy(fusion_score, text_score, image_score)
        unimodal_mean = 0.5 * (text_score + image_score)
        out = fusion_score.copy()
        mask = d > self.discrepancy_threshold
        out[mask] = (1 - self.refinement_strength) * fusion_score[mask] + self.refinement_strength * unimodal_mean[mask]
        return out

    def fit(self, text_out: AnalystOutput, image_out: AnalystOutput, labels: np.ndarray, num_classes: int, head_kwargs: dict):
        x = self.fuse_embedding(text_out.embedding, image_out.embedding)
        self.head = TorchHeadWrapper(in_dim=x.shape[1], out_dim=num_classes, **head_kwargs)
        self.head.fit(x, labels)
        return self.forward(text_out, image_out)

    def forward(self, text_out: AnalystOutput, image_out: AnalystOutput) -> AnalystOutput:
        x = self.fuse_embedding(text_out.embedding, image_out.embedding)
        fusion_score = self.head.predict_proba(x) if self.head else np.zeros_like(text_out.score)
        if text_out.score.size and image_out.score.size:
            fusion_score = self.refine(fusion_score, text_out.score, image_out.score)
        rationale = [f"discrepancy={d:.4f}" for d in self.discrepancy(fusion_score, text_out.score, image_out.score)]
        return AnalystOutput(embedding=x, score=fusion_score, rationale=rationale)
