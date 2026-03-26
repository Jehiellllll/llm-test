from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class SeedConfig(BaseModel):
    seed: int = 42


class TextConfig(BaseModel):
    max_features: int = 5000


class ImageConfig(BaseModel):
    bins: int = 16
    resize: int = 64
    video_sample_fps: int = 2
    max_video_frames: int = 8


class FusionConfig(BaseModel):
    discrepancy_threshold: float = 0.25
    refinement_strength: float = 0.3


class KBConfig(BaseModel):
    top_k: int = 5
    backend: Literal["faiss", "sklearn"] = "faiss"


class AggregatorConfig(BaseModel):
    alpha: float = 0.7
    beta: float = 0.3
    mode: Literal["weighted", "learned"] = "weighted"


class TrainConfig(BaseModel):
    epochs: int = 25
    lr: float = 1e-2
    weight_decay: float = 0.0
    hidden_dim: int = 128
    batch_size: int = 64


class ExperimentConfig(BaseModel):
    output_dir: str = "outputs"
    device: str = "cpu"


class PipelineConfig(BaseModel):
    seed: SeedConfig = Field(default_factory=SeedConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    kb: KBConfig = Field(default_factory=KBConfig)
    aggregator: AggregatorConfig = Field(default_factory=AggregatorConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)


def load_config(path: str | Path) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return PipelineConfig.model_validate(raw)
