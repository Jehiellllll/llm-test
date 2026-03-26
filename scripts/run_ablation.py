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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--output", default="outputs/ablation_results.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_ds = SentiMMDLikeDataset(args.train_jsonl)
    eval_ds = SentiMMDLikeDataset(args.eval_jsonl)

    y_true = np.array([LABEL_TO_ID[l] for l in eval_ds.labels()])

    results = {}
    for ab in ABLATIONS:
        flags = ModuleFlags.from_ablation(ab)
        model = SentiMMPipeline(cfg, flags=flags)
        model.fit(train_ds.texts(), train_ds.image_paths(), train_ds.kb_texts(), train_ds.labels())
        pred = model.predict(eval_ds.texts(), eval_ds.image_paths(), eval_ds.kb_texts())
        results[ab] = evaluate_metrics(y_true, pred)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
