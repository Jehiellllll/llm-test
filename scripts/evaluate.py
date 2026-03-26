#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sentimm_repro.data.dataset import SentiMMDLikeDataset
from sentimm_repro.labels import LABEL_TO_ID
from sentimm_repro.metrics import evaluate_metrics
from sentimm_repro.pipeline import SentiMMPipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--metrics-out", default="outputs/metrics.json")
    p.add_argument("--pred-out", default="outputs/predictions.json")
    args = p.parse_args()

    ds = SentiMMDLikeDataset(args.eval_jsonl)
    model = SentiMMPipeline.load(args.model)

    y_true = np.array([LABEL_TO_ID[x] for x in ds.labels()])
    score, aux = model.predict_proba(ds.texts(), ds.image_paths(), ds.video_paths(), ds.kb_texts())
    y_pred = score.argmax(axis=1)

    metrics = evaluate_metrics(y_true, y_pred)
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    model.save_predictions(score, args.pred_out)
    Path("outputs/retrieval_summary.json").write_text(json.dumps(aux, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
