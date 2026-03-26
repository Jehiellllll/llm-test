from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

from sentimm_repro.models.torch_head import TorchHeadWrapper
from sentimm_repro.types import AnalystOutput


class ImageAnalyst:
    def __init__(self, bins: int = 16, resize: int = 64, video_sample_fps: int = 2, max_video_frames: int = 8):
        self.bins = bins
        self.resize = (resize, resize)
        self.video_sample_fps = video_sample_fps
        self.max_video_frames = max_video_frames
        self.head: TorchHeadWrapper | None = None

    def _image_feat(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32) / 255.0
        feats = []
        for c in range(3):
            hist, _ = np.histogram(arr[:, :, c], bins=self.bins, range=(0.0, 1.0), density=True)
            feats.extend(hist.tolist())
        feats.extend(arr.mean(axis=(0, 1)).tolist())
        feats.extend(arr.std(axis=(0, 1)).tolist())
        return np.asarray(feats, dtype=np.float32)

    def _read_image(self, path: str | None) -> np.ndarray:
        if not path or not Path(path).exists():
            return np.zeros(self.bins * 3 + 6, dtype=np.float32)
        arr = np.asarray(Image.open(path).convert("RGB").resize(self.resize))
        return self._image_feat(arr)

    def _read_video(self, path: str | None) -> np.ndarray:
        if not path or not Path(path).exists():
            return np.zeros(self.bins * 3 + 6, dtype=np.float32)
        try:
            frames = iio.imiter(path)
            feats = []
            for i, frame in enumerate(frames):
                if i % max(self.video_sample_fps, 1) != 0:
                    continue
                img = Image.fromarray(frame).convert("RGB").resize(self.resize)
                feats.append(self._image_feat(np.asarray(img)))
                if len(feats) >= self.max_video_frames:
                    break
            if not feats:
                return np.zeros(self.bins * 3 + 6, dtype=np.float32)
            return np.mean(np.vstack(feats), axis=0)
        except Exception:
            return np.zeros(self.bins * 3 + 6, dtype=np.float32)

    def extract(self, image_paths: list[str | None], video_paths: list[str | None]) -> np.ndarray:
        feats = []
        for img, vid in zip(image_paths, video_paths):
            if vid:
                feats.append(self._read_video(vid))
            else:
                feats.append(self._read_image(img))
        return np.vstack(feats)

    def fit(self, image_paths: list[str | None], video_paths: list[str | None], labels: np.ndarray, num_classes: int, head_kwargs: dict):
        x = self.extract(image_paths, video_paths)
        self.head = TorchHeadWrapper(in_dim=x.shape[1], out_dim=num_classes, **head_kwargs)
        self.head.fit(x, labels)
        return self.forward(image_paths, video_paths)

    def forward(self, image_paths: list[str | None], video_paths: list[str | None]) -> AnalystOutput:
        x = self.extract(image_paths, video_paths)
        score = self.head.predict_proba(x) if self.head else np.zeros((len(image_paths), 7), dtype=np.float32)
        rationale = ["video" if v else "image" for v in video_paths]
        return AnalystOutput(embedding=x, score=score, rationale=rationale)
