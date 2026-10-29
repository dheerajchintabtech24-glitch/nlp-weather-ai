"""Sentence-level dialect identification with character n-gram features."""

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from ..preprocessing import detect_language, normalize_text


def build_features():
    return FeatureUnion(
        [
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=2, sublinear_tf=True)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+", sublinear_tf=True)),
        ]
    )


CLASSIFIERS = {
    "svm": lambda: LinearSVC(C=0.5),
    "logreg": lambda: LogisticRegression(max_iter=3000, C=5.0),
    "nb": lambda: MultinomialNB(alpha=0.1),
}


class DialectIdentifier:
    """Predicts labels like ``ta:kongu``; scores are restricted to the script-detected language."""

    def __init__(self, classifier: str = "svm"):
        self.classifier = classifier
        self.pipeline = Pipeline([("features", build_features()), ("clf", CLASSIFIERS[classifier]())])

    def fit(self, texts, labels):
        self.pipeline.fit([normalize_text(t) for t in texts], labels)
        return self

    @property
    def labels(self):
        return list(self.pipeline.classes_)

    def scores(self, texts):
        texts = [normalize_text(t) for t in texts]
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            return self.pipeline.predict_proba(texts)
        z = self.pipeline.decision_function(texts)
        z = np.exp(z - z.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)

    def predict(self, texts, restrict_to_script: bool = True):
        scores = self.scores(texts)
        labels = self.labels
        out = []
        for text, row in zip(texts, scores):
            row = row.copy()
            if restrict_to_script:
                lang, _ = detect_language(text)
                mask = np.array([lab.split(":")[0] == lang for lab in labels])
                if mask.any():
                    row = np.where(mask, row, 0.0)
                    row = row / row.sum() if row.sum() > 0 else row
            k = int(row.argmax())
            out.append((labels[k], float(row[k])))
        return out

    def predict_one(self, text):
        return self.predict([text])[0]

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "DialectIdentifier":
        return joblib.load(path)
