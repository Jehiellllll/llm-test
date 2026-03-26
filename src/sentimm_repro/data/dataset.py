from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sentimm_repro.labels import LABEL_TO_ID


@dataclass
class Sample:
    sample_id: str
    text: str
    image_path: str | None
    label: str
    kb_text: str


class SentiMMDLikeDataset:
    """Task-compatible dataset interface for SentiMMD-style records.

    Expected JSONL schema per line:
    {
      "id": "...",
      "text": "...",
      "image_path": "relative/or/absolute/path.jpg",  # optional
      "label": "Like|Happiness|Anger|Disgust|Fear|Sadness|Surprise",
      "kb_text": "optional external knowledge text"
    }
    """

    def __init__(self, jsonl_path: str | Path, root_dir: str | Path | None = None):
        self.jsonl_path = Path(jsonl_path)
        self.root_dir = Path(root_dir) if root_dir else self.jsonl_path.parent
        self.samples = list(self._load())

    def _load(self) -> Iterable[Sample]:
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                label = row["label"]
                if label not in LABEL_TO_ID:
                    raise ValueError(f"Unknown label: {label}")
                image_path = row.get("image_path")
                if image_path:
                    img = Path(image_path)
                    if not img.is_absolute():
                        image_path = str((self.root_dir / img).resolve())
                yield Sample(
                    sample_id=str(row.get("id", "")),
                    text=row.get("text", ""),
                    image_path=image_path,
                    label=label,
                    kb_text=row.get("kb_text", ""),
                )

    def texts(self) -> list[str]:
        return [s.text for s in self.samples]

    def kb_texts(self) -> list[str]:
        return [s.kb_text for s in self.samples]

    def image_paths(self) -> list[str | None]:
        return [s.image_path for s in self.samples]

    def labels(self) -> list[str]:
        return [s.label for s in self.samples]
