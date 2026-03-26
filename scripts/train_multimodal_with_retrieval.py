#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentimm_repro.config import load_config
from sentimm_repro.data.dataset import SentiMMDLikeDataset
from sentimm_repro.pipeline import ModuleFlags, SentiMMPipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--model-out", default="outputs/multimodal_retrieval.joblib")
    args = p.parse_args()

    cfg = load_config(args.config)
    ds = SentiMMDLikeDataset(args.train_jsonl)
    flags = ModuleFlags(text=True, image=True, fusion=True, kb=True, aggregator=True)

    model = SentiMMPipeline(cfg, flags)
    model.fit(ds.texts(), ds.image_paths(), ds.video_paths(), ds.kb_texts(), ds.labels())
    model.save(args.model_out)

    log = {"mode": "multimodal_with_retrieval", "train_size": len(ds.samples), "model_out": args.model_out}
    Path("outputs/train_multimodal_with_retrieval_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(log)


if __name__ == "__main__":
    main()
