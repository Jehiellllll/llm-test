from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class ClassifierAggregator:
    def __init__(self, c: float = 1.0, random_state: int = 42):
        self.model = LogisticRegression(
            C=c,
            max_iter=1000,
            multi_class="multinomial",
            random_state=random_state,
        )

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.model.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)
