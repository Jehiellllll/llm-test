from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }


def dump_metrics(metrics: dict, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
