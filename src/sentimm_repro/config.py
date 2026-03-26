from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PipelineConfig:
    random_state: int = 42
    text_max_features: int = 5000
    kb_max_features: int = 2000
    image_bins: int = 16
    classifier_c: float = 1.0
    output_dir: str = "outputs"
    ablations: list[str] = field(default_factory=list)



def load_config(path: str | Path) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return PipelineConfig(**raw)
