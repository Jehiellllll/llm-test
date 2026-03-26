from __future__ import annotations

import json

import numpy as np
from PIL import Image

from sentimm_repro.config import PipelineConfig
from sentimm_repro.data.dataset import SentiMMDLikeDataset
from sentimm_repro.labels import EMOTIONS
from sentimm_repro.pipeline import ModuleFlags, SentiMMPipeline


def _mk_img(path, seed):
    rng = np.random.default_rng(seed)
    arr = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _make_ds(tmp_path):
    rows = []
    for i, label in enumerate(EMOTIONS):
        img = tmp_path / f"{i}.png"
        _mk_img(img, i)
        rows.append({"id": str(i), "text": f"text {label}", "image_path": str(img), "video_path": None, "label": label, "kb_text": f"kb {label}", "metadata": {}})
    fp = tmp_path / "data.jsonl"
    fp.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")
    return SentiMMDLikeDataset(fp)


def test_full_pipeline(tmp_path):
    ds = _make_ds(tmp_path)
    model = SentiMMPipeline(PipelineConfig(), ModuleFlags.from_ablation("full"))
    model.fit(ds.texts(), ds.image_paths(), ds.video_paths(), ds.kb_texts(), ds.labels())
    pred = model.predict(ds.texts(), ds.image_paths(), ds.video_paths(), ds.kb_texts())
    assert pred.shape[0] == len(ds.samples)


def test_ablation_no_kb(tmp_path):
    ds = _make_ds(tmp_path)
    model = SentiMMPipeline(PipelineConfig(), ModuleFlags.from_ablation("no KB Assistant"))
    model.fit(ds.texts(), ds.image_paths(), ds.video_paths(), ds.kb_texts(), ds.labels())
    pred = model.predict(ds.texts(), ds.image_paths(), ds.video_paths(), ds.kb_texts())
    assert pred.shape[0] == len(ds.samples)
