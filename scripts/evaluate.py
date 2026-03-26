#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from sentimm_repro.data.dataset import SentiMMDLikeDataset
from sentimm_repro.labels import LABEL_TO_ID
from sentimm_repro.metrics import dump_metrics, evaluate_metrics
from sentimm_repro.pipeline import SentiMMPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--metrics-out", default="outputs/metrics.json")
    args = parser.parse_args()

    ds = SentiMMDLikeDataset(args.eval_jsonl)
    model = SentiMMPipeline.load(args.model)

    y_true = np.array([LABEL_TO_ID[l] for l in ds.labels()])
    y_pred = model.predict(ds.texts(), ds.image_paths(), ds.kb_texts())

    metrics = evaluate_metrics(y_true, y_pred)
    dump_metrics(metrics, args.metrics_out)
    print(metrics)


if __name__ == "__main__":
    main()
