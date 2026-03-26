#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sentimm_repro.config import load_config
from sentimm_repro.data.dataset import SentiMMDLikeDataset
from sentimm_repro.pipeline import ModuleFlags, SentiMMPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--ablation", default="full")
    parser.add_argument("--model-out", default="outputs/model.joblib")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = SentiMMDLikeDataset(args.train_jsonl)
    flags = ModuleFlags.from_ablation(args.ablation)

    model = SentiMMPipeline(cfg, flags=flags)
    model.fit(ds.texts(), ds.image_paths(), ds.kb_texts(), ds.labels())
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)
    print(f"saved model: {args.model_out}")


if __name__ == "__main__":
    main()
