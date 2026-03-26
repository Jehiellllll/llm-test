from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


class KBAssistant:
    def __init__(self, max_features: int = 2000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
            strip_accents="unicode",
        )

    def fit_transform(self, kb_texts: list[str]):
        return self.vectorizer.fit_transform(kb_texts)

    def transform(self, kb_texts: list[str]):
        return self.vectorizer.transform(kb_texts)
