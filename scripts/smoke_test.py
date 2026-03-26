#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from sentimm_repro.config import PipelineConfig
from sentimm_repro.data.dataset import SentiMMDLikeDataset
from sentimm_repro.labels import EMOTIONS
from sentimm_repro.pipeline import ModuleFlags, SentiMMPipeline


def _mk_img(path: Path, seed: int):
    rng = np.random.default_rng(seed)
    arr = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        train = root / "train.jsonl"
        rows = []
        for i, label in enumerate(EMOTIONS):
            img = root / f"img_{i}.png"
            _mk_img(img, i)
            rows.append({"id": str(i), "text": f"sample {label}", "image_path": str(img), "video_path": None, "kb_text": f"knowledge {label}", "label": label, "metadata": {"split": "train"}})
        train.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

        ds = SentiMMDLikeDataset(train)
        model = SentiMMPipeline(PipelineConfig(), ModuleFlags.from_ablation("full"))
        model.fit(ds.texts(), ds.image_paths(), ds.video_paths(), ds.kb_texts(), ds.labels())
        preds = model.predict(ds.texts(), ds.image_paths(), ds.video_paths(), ds.kb_texts())
        assert len(preds) == len(rows)
        print("smoke passed")


if __name__ == "__main__":
    main()
