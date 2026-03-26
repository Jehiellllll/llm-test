from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AnalystOutput:
    embedding: np.ndarray
    score: np.ndarray
    rationale: list[str] | None = None
