"""
layer1_nlp_parser.py
──────────────────────────────────────────────────────────────
AI LAYER 1: NLP Invoice Text Classifier
AI TYPE: TF-IDF Vectorizer + Gradient Boosting Classifier
──────────────────────────────────────────────────────────────
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import joblib


class NLPEmailClassifier:
    LABEL_NAMES = {0: "NORMAL", 1: "URGENT", 2: "SUSPICIOUS"}

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=500, sublinear_tf=True, stop_words="english")),
            ("clf",   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
        ])
        self.classes_   = []
        self.is_fitted  = False

    def fit(self, texts: list, labels: list) -> "NLPEmailClassifier":
        self.pipeline.fit(texts, labels)
        self.classes_   = sorted(set(labels))
        self.is_fitted  = True
        return self

    def predict(self, text: str) -> dict:
        proba = self.pipeline.predict_proba([text])[0]
        # Map to full 3-class risk weights regardless of classes found in training
        # classes_ holds the actual class indices present during training
        risk_map = {0: 0.0, 1: 0.4, 2: 1.0}
        label_idx = int(np.argmax(proba))
        actual_label = int(self.classes_[label_idx]) if self.classes_ else label_idx
        conf  = float(proba[label_idx])

        # Build full risk score across observed classes
        risk_score = 0.0
        for i, cls in enumerate(self.classes_):
            risk_score += float(proba[i]) * risk_map.get(int(cls), 0.5)

        label_name = self.LABEL_NAMES.get(actual_label, "NORMAL")

        return {
            "label":      actual_label,
            "label_name": label_name,
            "confidence": round(conf, 4),
            "risk_score": round(risk_score, 4),
            "probabilities": {k: round(float(proba[i]), 4) if i < len(proba) else 0.0
                              for i, k in enumerate(["NORMAL","URGENT","SUSPICIOUS"])
                              if i < len(proba)},
        }

    def save(self, path: str):
        joblib.dump({"pipeline": self.pipeline, "classes": self.classes_}, path)

    def load(self, path: str) -> "NLPEmailClassifier":
        d = joblib.load(path)
        self.pipeline  = d["pipeline"]
        self.classes_  = d["classes"]
        self.is_fitted = True
        return self
