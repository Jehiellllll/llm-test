from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from sentimm_repro.agents.classifier_aggregator import ClassifierAggregator
from sentimm_repro.agents.fusion_inspector import FusionInspector
from sentimm_repro.agents.image_analyst import ImageAnalyst
from sentimm_repro.agents.kb_assistant import KBAssistant
from sentimm_repro.agents.text_analyst import TextAnalyst
from sentimm_repro.config import PipelineConfig
from sentimm_repro.labels import LABEL_TO_ID


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
        elif name in ("none", "full", ""):
            pass
        else:
            raise ValueError(f"Unsupported ablation: {name}")
        return flags


class SentiMMPipeline:
    def __init__(self, config: PipelineConfig, flags: ModuleFlags | None = None):
        self.config = config
        self.flags = flags or ModuleFlags()

        self.text_analyst = TextAnalyst(max_features=config.text_max_features)
        self.image_analyst = ImageAnalyst(bins=config.image_bins)
        self.fusion_inspector = FusionInspector()
        self.kb_assistant = KBAssistant(max_features=config.kb_max_features)
        self.classifier_aggregator = ClassifierAggregator(c=config.classifier_c, random_state=config.random_state)

        self.base_heads: dict[str, LogisticRegression] = {}
        self._trained = False

    @staticmethod
    def _dense(x):
        if sparse.issparse(x):
            return x.toarray()
        return x

    @staticmethod
    def _concat(features: list[np.ndarray]) -> np.ndarray:
        dense = [SentiMMPipeline._dense(f) for f in features if f is not None]
        if not dense:
            raise ValueError("No active features. At least one module must remain enabled.")
        return np.concatenate(dense, axis=1)

    def _train_head(self, x, y):
        model = LogisticRegression(
            C=self.config.classifier_c,
            max_iter=1000,
            multi_class="multinomial",
            random_state=self.config.random_state,
        )
        model.fit(self._dense(x), y)
        return model

    def fit(self, texts: list[str], image_paths: list[str | None], kb_texts: list[str], labels: list[str]):
        y = np.array([LABEL_TO_ID[l] for l in labels], dtype=np.int64)
        heads_train_probs = []
        raw_features = []

        if self.flags.text:
            text_x = self.text_analyst.fit_transform(texts)
            text_head = self._train_head(text_x, y)
            self.base_heads["text"] = text_head
            heads_train_probs.append(text_head.predict_proba(self._dense(text_x)))
            raw_features.append(self._dense(text_x))
        else:
            text_x = None

        if self.flags.image:
            image_x = self.image_analyst.fit_transform(image_paths)
            img_head = self._train_head(image_x, y)
            self.base_heads["image"] = img_head
            heads_train_probs.append(img_head.predict_proba(image_x))
            raw_features.append(image_x)
        else:
            image_x = None

        if self.flags.kb:
            kb_x = self.kb_assistant.fit_transform(kb_texts)
            kb_head = self._train_head(kb_x, y)
            self.base_heads["kb"] = kb_head
            heads_train_probs.append(kb_head.predict_proba(self._dense(kb_x)))
            raw_features.append(self._dense(kb_x))
        else:
            kb_x = None

        if self.flags.fusion and text_x is not None and image_x is not None:
            fusion_x = self.fusion_inspector.fit_transform(text_x, image_x)
            fusion_head = self._train_head(fusion_x, y)
            self.base_heads["fusion"] = fusion_head
            heads_train_probs.append(fusion_head.predict_proba(fusion_x))
            raw_features.append(fusion_x)

        if self.flags.aggregator:
            agg_x = self._concat(heads_train_probs)
            self.classifier_aggregator.fit(agg_x, y)
        else:
            self.classifier_aggregator = None

        self._trained = True
        return self

    def _forward_heads(self, texts, image_paths, kb_texts):
        probs = []
        if "text" in self.base_heads:
            text_x = self.text_analyst.transform(texts)
            probs.append(self.base_heads["text"].predict_proba(self._dense(text_x)))
        else:
            text_x = None

        if "image" in self.base_heads:
            image_x = self.image_analyst.transform(image_paths)
            probs.append(self.base_heads["image"].predict_proba(image_x))
        else:
            image_x = None

        if "kb" in self.base_heads:
            kb_x = self.kb_assistant.transform(kb_texts)
            probs.append(self.base_heads["kb"].predict_proba(self._dense(kb_x)))
        else:
            kb_x = None

        if "fusion" in self.base_heads and text_x is not None and image_x is not None:
            fusion_x = self.fusion_inspector.transform(text_x, image_x)
            probs.append(self.base_heads["fusion"].predict_proba(fusion_x))

        if not probs:
            raise RuntimeError("No trained heads found.")
        return probs

    def predict(self, texts: list[str], image_paths: list[str | None], kb_texts: list[str]) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Pipeline is not trained.")

        probs = self._forward_heads(texts, image_paths, kb_texts)
        if self.classifier_aggregator is not None:
            x = self._concat(probs)
            return self.classifier_aggregator.predict(x)

        mean_prob = np.mean(np.stack(probs, axis=0), axis=0)
        return np.argmax(mean_prob, axis=1)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "SentiMMPipeline":
        return joblib.load(path)
