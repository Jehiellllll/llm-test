from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sentimm_repro.labels import LABEL_TO_ID


@dataclass
class Sample:
    sample_id: str
    text: str
    image_path: str | None
    video_path: str | None
    label: str
    kb_text: str
    metadata: dict[str, Any]


class SentiMMDLikeDataset:
    """Task-compatible schema for framework reproduction.

    JSONL row:
    {"id":"...","text":"...","image_path":"...","video_path":"...",
     "label":"Like|...","kb_text":"...","metadata":{...}}
    """

    def __init__(self, jsonl_path: str | Path, root_dir: str | Path | None = None):
        self.jsonl_path = Path(jsonl_path)
        self.root_dir = Path(root_dir) if root_dir else self.jsonl_path.parent
        self.samples = list(self._load())

    def _resolve_path(self, value: str | None) -> str | None:
        if not value:
            return None
        p = Path(value)
        if p.is_absolute():
            return str(p)
        return str((self.root_dir / p).resolve())

    def _load(self) -> Iterable[Sample]:
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                label = row["label"]
                if label not in LABEL_TO_ID:
                    raise ValueError(f"Unknown label: {label}")
                yield Sample(
                    sample_id=str(row.get("id", "")),
                    text=row.get("text", ""),
                    image_path=self._resolve_path(row.get("image_path")),
                    video_path=self._resolve_path(row.get("video_path")),
                    label=label,
                    kb_text=row.get("kb_text", ""),
                    metadata=row.get("metadata", {}),
                )

    def ids(self) -> list[str]:
        return [s.sample_id for s in self.samples]

    def texts(self) -> list[str]:
        return [s.text for s in self.samples]

    def kb_texts(self) -> list[str]:
        return [s.kb_text for s in self.samples]

    def image_paths(self) -> list[str | None]:
        return [s.image_path for s in self.samples]

    def video_paths(self) -> list[str | None]:
        return [s.video_path for s in self.samples]

    def labels(self) -> list[str]:
        return [s.label for s in self.samples]
