from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class ImageAnalyst:
    def __init__(self, bins: int = 16, resize: tuple[int, int] = (64, 64)):
        self.bins = bins
        self.resize = resize

    def _one(self, path: str | None) -> np.ndarray:
        if not path or not Path(path).exists():
            return np.zeros(self.bins * 3 + 6, dtype=np.float32)

        img = Image.open(path).convert("RGB").resize(self.resize)
        arr = np.asarray(img, dtype=np.float32) / 255.0

        feats = []
        for c in range(3):
            hist, _ = np.histogram(arr[:, :, c], bins=self.bins, range=(0.0, 1.0), density=True)
            feats.extend(hist.tolist())

        channel_mean = arr.mean(axis=(0, 1))
        channel_std = arr.std(axis=(0, 1))
        feats.extend(channel_mean.tolist())
        feats.extend(channel_std.tolist())
        return np.asarray(feats, dtype=np.float32)

    def fit_transform(self, image_paths: list[str | None]) -> np.ndarray:
        return np.vstack([self._one(p) for p in image_paths])

    def transform(self, image_paths: list[str | None]) -> np.ndarray:
        return np.vstack([self._one(p) for p in image_paths])
