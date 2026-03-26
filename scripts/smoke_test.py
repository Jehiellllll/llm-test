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
from sentimm_repro.pipeline import SentiMMPipeline


def _mk_img(path: Path, seed: int):
    rng = np.random.default_rng(seed)
    arr = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        train = root / "train.jsonl"
        test = root / "test.jsonl"
        rows = []
        for i, label in enumerate(EMOTIONS):
            p = root / f"img_{i}.png"
            _mk_img(p, i)
            rows.append({"id": f"{i}", "text": f"sample text {label}", "image_path": str(p), "kb_text": f"kb {label}", "label": label})
        train.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        test.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

        ds_train = SentiMMDLikeDataset(train)
        ds_test = SentiMMDLikeDataset(test)
        model = SentiMMPipeline(PipelineConfig())
        model.fit(ds_train.texts(), ds_train.image_paths(), ds_train.kb_texts(), ds_train.labels())
        pred = model.predict(ds_test.texts(), ds_test.image_paths(), ds_test.kb_texts())
        assert pred.shape[0] == len(rows)
        print("smoke test passed")


if __name__ == "__main__":
    main()
