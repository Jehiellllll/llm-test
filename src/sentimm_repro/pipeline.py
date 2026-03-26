from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from sentimm_repro.agents.classifier_aggregator import ClassifierAggregator
from sentimm_repro.agents.fusion_inspector import FusionInspector
from sentimm_repro.agents.image_analyst import ImageAnalyst
from sentimm_repro.agents.kb_assistant import KBAssistant
from sentimm_repro.agents.text_analyst import TextAnalyst
from sentimm_repro.config import PipelineConfig
from sentimm_repro.labels import LABEL_TO_ID
from sentimm_repro.utils.seed import set_seed


@dataclass
class ModuleFlags:
    text: bool = True
    image: bool = True
    fusion: bool = True
    kb: bool = True
    aggregator: bool = True

    @classmethod
    def from_ablation(cls, name: str):
        flags = cls()
        if name == "no KB Assistant":
            flags.kb = False
        elif name == "no Fusion Inspector":
            flags.fusion = False
        elif name == "no Image Analyst":
            flags.image = False
        elif name == "no Text Analyst":
            flags.text = False
        elif name == "no Classifier Aggregator":
            flags.aggregator = False
        elif name in ("", "full", "none"):
            pass
        else:
            raise ValueError(f"Unsupported ablation: {name}")
        return flags


class SentiMMPipeline:
    def __init__(self, config: PipelineConfig, flags: ModuleFlags | None = None):
        self.config = config
        self.flags = flags or ModuleFlags()
        self.num_classes = 7

        self.text_analyst = TextAnalyst(max_features=config.text.max_features)
        self.image_analyst = ImageAnalyst(
            bins=config.image.bins,
            resize=config.image.resize,
            video_sample_fps=config.image.video_sample_fps,
            max_video_frames=config.image.max_video_frames,
        )
        self.fusion_inspector = FusionInspector(
            discrepancy_threshold=config.fusion.discrepancy_threshold,
            refinement_strength=config.fusion.refinement_strength,
        )
        self.kb_assistant = KBAssistant(top_k=config.kb.top_k, backend=config.kb.backend)
        self.aggregator = ClassifierAggregator(
            alpha=config.aggregator.alpha,
            beta=config.aggregator.beta,
            mode=config.aggregator.mode,
        )
        self._trained = False

    def _head_kwargs(self):
        return {
            "hidden_dim": self.config.train.hidden_dim,
            "lr": self.config.train.lr,
            "epochs": self.config.train.epochs,
            "batch_size": self.config.train.batch_size,
            "device": self.config.experiment.device,
        }

    def fit(self, texts, image_paths, video_paths, kb_texts, labels):
        set_seed(self.config.seed.seed)
        y = np.array([LABEL_TO_ID[l] for l in labels], dtype=np.int64)

        text_out = None
        image_out = None
        fusion_out = None

        if self.flags.text:
            text_out = self.text_analyst.fit(texts, y, self.num_classes, self._head_kwargs())
        if self.flags.image:
            image_out = self.image_analyst.fit(image_paths, video_paths, y, self.num_classes, self._head_kwargs())

        if self.flags.fusion and text_out is not None and image_out is not None:
            fusion_out = self.fusion_inspector.fit(text_out, image_out, y, self.num_classes, self._head_kwargs())
        elif text_out is not None and image_out is not None:
            mean_score = 0.5 * (text_out.score + image_out.score)
            fusion_out = text_out
            fusion_out.score = mean_score
            fusion_out.embedding = np.concatenate([text_out.embedding, image_out.embedding], axis=1)
        elif text_out is not None:
            fusion_out = text_out
        elif image_out is not None:
            fusion_out = image_out
        else:
            raise ValueError("At least one of Text/Image analyst must be active")

        if self.flags.kb:
            self.kb_assistant.build(kb_texts, y)

        retrieved_score = np.zeros_like(fusion_out.score)
        if self.flags.kb:
            _, kb_result = self.kb_assistant.query(kb_texts, num_classes=self.num_classes)
            retrieved_score = kb_result.label_distribution

        if self.flags.aggregator:
            self.aggregator.fit(fusion_out.score, retrieved_score, y, head_kwargs=self._head_kwargs())

        self._trained = True
        return self

    def predict_proba(self, texts, image_paths, video_paths, kb_texts):
        if not self._trained:
            raise RuntimeError("Pipeline not trained")

        text_out = self.text_analyst.forward(texts) if self.flags.text else None
        image_out = self.image_analyst.forward(image_paths, video_paths) if self.flags.image else None

        if self.flags.fusion and text_out is not None and image_out is not None:
            fusion_out = self.fusion_inspector.forward(text_out, image_out)
        elif text_out is not None and image_out is not None:
            fusion_out = text_out
            fusion_out.score = 0.5 * (text_out.score + image_out.score)
        elif text_out is not None:
            fusion_out = text_out
        elif image_out is not None:
            fusion_out = image_out
        else:
            raise ValueError("At least one of Text/Image analyst must be active")

        retrieved_score = np.zeros_like(fusion_out.score)
        retrieval_summary = {"indices": [], "distances": []}
        if self.flags.kb:
            _, kb_result = self.kb_assistant.query(kb_texts, num_classes=self.num_classes)
            retrieved_score = kb_result.label_distribution
            retrieval_summary = {"indices": kb_result.indices.tolist(), "distances": kb_result.distances.tolist()}

        if self.flags.aggregator:
            final_score = self.aggregator.combine(fusion_out.score, retrieved_score)
        else:
            final_score = fusion_out.score

        return final_score, {
            "fusion_rationale": fusion_out.rationale,
            "retrieval_summary": retrieval_summary,
        }

    def predict(self, texts, image_paths, video_paths, kb_texts):
        score, _ = self.predict_proba(texts, image_paths, video_paths, kb_texts)
        return score.argmax(axis=1)

    def save_predictions(self, scores: np.ndarray, output_path: str | Path):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pred = [{"pred": int(row.argmax()), "score": row.tolist()} for row in scores]
        path.write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path):
        return joblib.load(path)
