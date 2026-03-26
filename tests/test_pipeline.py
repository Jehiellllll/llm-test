from __future__ import annotations

import json

import numpy as np
from PIL import Image

from sentimm_repro.config import PipelineConfig
from sentimm_repro.data.dataset import SentiMMDLikeDataset
from sentimm_repro.labels import EMOTIONS
from sentimm_repro.metrics import evaluate_metrics
from sentimm_repro.pipeline import ModuleFlags, SentiMMPipeline


def _mk_img(path, seed):
    rng = np.random.default_rng(seed)
    arr = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _build_jsonl(tmp_path):
    fp = tmp_path / "data.jsonl"
    rows = []
    for i, label in enumerate(EMOTIONS):
        img = tmp_path / f"{i}.png"
        _mk_img(img, i)
        rows.append({"id": str(i), "text": f"text {label}", "image_path": str(img), "kb_text": f"kb {label}", "label": label})
    fp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return fp


def test_pipeline_train_predict(tmp_path):
    data_file = _build_jsonl(tmp_path)
    ds = SentiMMDLikeDataset(data_file)

    model = SentiMMPipeline(PipelineConfig(), ModuleFlags.from_ablation("full"))
    model.fit(ds.texts(), ds.image_paths(), ds.kb_texts(), ds.labels())
    pred = model.predict(ds.texts(), ds.image_paths(), ds.kb_texts())

    assert pred.shape[0] == len(ds.samples)
    m = evaluate_metrics(np.arange(len(EMOTIONS)), pred)
    assert "macro_f1" in m


def test_ablation_no_aggregator(tmp_path):
    data_file = _build_jsonl(tmp_path)
    ds = SentiMMDLikeDataset(data_file)

    model = SentiMMPipeline(PipelineConfig(), ModuleFlags.from_ablation("no Classifier Aggregator"))
    model.fit(ds.texts(), ds.image_paths(), ds.kb_texts(), ds.labels())
    pred = model.predict(ds.texts(), ds.image_paths(), ds.kb_texts())
    assert pred.shape[0] == len(ds.samples)
