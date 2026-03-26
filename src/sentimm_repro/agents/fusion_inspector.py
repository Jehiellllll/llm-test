from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler


class FusionInspector:
    """Builds hand-crafted cross-modal consistency features."""

    def __init__(self):
        self.scaler = StandardScaler()

    @staticmethod
    def _dense(x):
        if sparse.issparse(x):
            return x.toarray()
        return x

    def _make_features(self, text_x, image_x) -> np.ndarray:
        t = self._dense(text_x)
        i = self._dense(image_x)

        t_mean = t.mean(axis=1)
        i_mean = i.mean(axis=1)
        t_norm = np.linalg.norm(t, axis=1)
        i_norm = np.linalg.norm(i, axis=1)
        dot = (t[:, : min(t.shape[1], i.shape[1])] * i[:, : min(t.shape[1], i.shape[1])]).sum(axis=1)
        cos = dot / (t_norm * i_norm + 1e-8)
        diff = np.abs(t_mean - i_mean)

        return np.vstack([t_mean, i_mean, t_norm, i_norm, cos, diff]).T

    def fit_transform(self, text_x, image_x) -> np.ndarray:
        f = self._make_features(text_x, image_x)
        return self.scaler.fit_transform(f)

    def transform(self, text_x, image_x) -> np.ndarray:
        f = self._make_features(text_x, image_x)
        return self.scaler.transform(f)
