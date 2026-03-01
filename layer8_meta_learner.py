"""
layer8_meta_learner.py
──────────────────────────────────────────────────────────────
AI LAYER 8: Meta-Learner (Stacked Ensemble Decision Engine)
──────────────────────────────────────────────────────────────
PURPOSE: Take ALL risk signals from Layers 1–7 and make the
         FINAL decision using a trained meta-classifier.

AI TYPE: Random Forest Meta-Classifier (stacking)
  - Input: risk scores from all 7 preceding AI layers
  - Output: FINAL risk score + AUTO_POST / REVIEW / HOLD
  - Trained on ABC's historical labelled outcomes

WHY A META-LEARNER IS MORE POWERFUL:
  Simple approach: take average of all scores.
  Meta-learner approach: LEARNS that for Logistics invoices,
  the Vendor Profiler score matters more; for new vendors,
  the NLP score matters more; etc.

  The meta-learner learns CROSS-SIGNAL INTERACTIONS that a
  simple average completely misses.
──────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib


# Layer risk signal columns (input to meta-learner)
META_FEATURES = [
    "nlp_risk",           # Layer 1
    "dedup_risk",         # Layer 2
    "iso_forest_risk",    # Layer 3
    "dbscan_risk",        # Layer 4
    "mlp_risk",           # Layer 5
    "vendor_profile_risk",# Layer 6
    "po_tolerance_risk",  # Layer 7
    # Interaction features (AI-engineered)
    "nlp_x_ensemble",     # NLP × ensemble average (interaction)
    "vendor_x_po",        # vendor_profile × PO tolerance (interaction)
    "max_signal",         # Max of all 7 signals
    "signals_above_05",   # Count of signals > 0.5
]

# Decision thresholds
THRESHOLDS = {"AUTO_POST": 0.28, "HOLD": 0.58}


class MetaLearner:
    """
    AI Layer 8: Stacked ensemble final decision maker.
    Combines all layer signals with a trained Random Forest meta-classifier.
    """

    def __init__(self):
        base_rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=3,
            class_weight={0: 1, 1: 10},   # Penalize missed anomalies 10×
            random_state=42,
        )
        # Calibrate probabilities for reliable confidence scores
        self.clf = CalibratedClassifierCV(base_rf, cv=3, method="isotonic")
        self.is_fitted = False

    def _build_meta_features(self, signals: dict) -> np.ndarray:
        """Build meta-feature vector from layer signals."""
        s1  = signals.get("nlp_risk", 0)
        s2  = signals.get("dedup_risk", 0)
        s3  = signals.get("iso_forest_risk", 0)
        s4  = signals.get("dbscan_risk", 0)
        s5  = signals.get("mlp_risk", 0)
        s6  = signals.get("vendor_profile_risk", 0)
        s7  = signals.get("po_tolerance_risk", 0)

        all_scores = [s1, s2, s3, s4, s5, s6, s7]
        ens_avg    = np.mean([s3, s4, s5])   # ensemble average (layers 3-5)

        return np.array([[
            s1, s2, s3, s4, s5, s6, s7,
            s1 * ens_avg,          # nlp × ensemble
            s6 * s7,               # vendor × PO
            float(np.max(all_scores)),  # max signal
            float(sum(1 for s in all_scores if s > 0.5)),  # high-signal count
        ]])

    def fit(self, signal_rows: list[dict], labels: list[int]) -> "MetaLearner":
        """
        Train meta-learner on collected layer signals + true labels.
        signal_rows: list of signal dicts from all 7 layers
        labels: 0=normal, 1=anomaly
        """
        X = np.vstack([self._build_meta_features(r) for r in signal_rows])
        y = np.array(labels)

        if len(X) >= 20 and len(set(y)) > 1:
            self.clf.fit(X, y)
            self.is_fitted = True
        return self

    def predict(self, signals: dict) -> dict:
        """
        Make final decision from all layer signals.
        Returns risk_score + decision + explanation.
        """
        X = self._build_meta_features(signals)

        if self.is_fitted:
            proba      = self.clf.predict_proba(X)[0]
            risk_score = float(proba[1])  # P(anomaly)
        else:
            # Fallback: weighted average of signals
            weights    = [0.10, 0.15, 0.18, 0.12, 0.15, 0.18, 0.12]
            signal_vals = [
                signals.get(k, 0) for k in
                ["nlp_risk","dedup_risk","iso_forest_risk","dbscan_risk",
                 "mlp_risk","vendor_profile_risk","po_tolerance_risk"]
            ]
            risk_score = float(np.dot(weights, signal_vals))

        risk_score = round(risk_score, 4)

        # Decision
        if risk_score < THRESHOLDS["AUTO_POST"]:
            decision = "AUTO_POST"
        elif risk_score < THRESHOLDS["HOLD"]:
            decision = "REVIEW"
        else:
            decision = "HOLD"

        # Which signals drove the decision?
        signal_items = [
            ("NLP Email",          signals.get("nlp_risk", 0)),
            ("Ensemble Anomaly",   max(signals.get("iso_forest_risk",0), signals.get("mlp_risk",0))),
            ("Vendor Profile",     signals.get("vendor_profile_risk", 0)),
            ("PO Tolerance",       signals.get("po_tolerance_risk", 0)),
        ]
        top_drivers = sorted(signal_items, key=lambda x: x[1], reverse=True)[:2]
        drivers_str = ", ".join(f"{n}={v:.2f}" for n,v in top_drivers if v > 0.1)

        return {
            "final_risk_score": risk_score,
            "risk_pct":         f"{risk_score*100:.1f}%",
            "decision":         decision,
            "confidence":       "HIGH" if abs(risk_score - 0.43) > 0.15 else "MEDIUM",
            "top_drivers":      drivers_str or "All signals low",
            "meta_model":       "RandomForest+Calibrated" if self.is_fitted else "WeightedFallback",
        }

    def save(self, path: str):
        joblib.dump(self.clf, path)

    def load(self, path: str) -> "MetaLearner":
        self.clf = joblib.load(path)
        self.is_fitted = True
        return self
