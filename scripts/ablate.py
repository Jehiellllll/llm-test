#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sentimm_repro.config import load_config
from sentimm_repro.data.dataset import SentiMMDLikeDataset
from sentimm_repro.labels import LABEL_TO_ID
from sentimm_repro.metrics import evaluate_metrics
from sentimm_repro.pipeline import ModuleFlags, SentiMMPipeline

ABLATIONS = [
    "full",
    "no KB Assistant",
    "no Fusion Inspector",
    "no Image Analyst",
    "no Text Analyst",
    "no Classifier Aggregator",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", default="outputs/ablation_results.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    train_ds = SentiMMDLikeDataset(args.train_jsonl)
    eval_ds = SentiMMDLikeDataset(args.eval_jsonl)
    y_true = np.array([LABEL_TO_ID[x] for x in eval_ds.labels()])

    out = {}
    for setting in ABLATIONS:
        flags = ModuleFlags.from_ablation(setting)
        model = SentiMMPipeline(cfg, flags)
        model.fit(train_ds.texts(), train_ds.image_paths(), train_ds.video_paths(), train_ds.kb_texts(), train_ds.labels())
        pred = model.predict(eval_ds.texts(), eval_ds.image_paths(), eval_ds.video_paths(), eval_ds.kb_texts())
        out[setting] = evaluate_metrics(y_true, pred)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
