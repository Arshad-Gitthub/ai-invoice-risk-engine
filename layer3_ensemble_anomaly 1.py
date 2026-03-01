"""
layer3_ensemble_anomaly.py
──────────────────────────────────────────────────────────────
AI LAYERS 3 + 4 + 5: Triple-Model Ensemble Anomaly Detector
──────────────────────────────────────────────────────────────
PURPOSE: Detect numeric anomalies using 3 independent AI models
         whose votes are combined — reducing false positives.

LAYER 3 — Isolation Forest (Unsupervised)
  Learns the density of "normal" invoice space.
  Fast, no labels needed. Best for amount/variance outliers.

LAYER 4 — DBSCAN Cluster Outlier (Unsupervised)
  Clusters normal invoices. Points that don't fit any cluster
  (noise points) are anomalies. Catches spatial outliers that
  Isolation Forest may miss (e.g. unusual quantity+price combos).

LAYER 5 — MLP Neural Network (Supervised, trained on labelled data)
  Once we have labelled anomalies in training data, the MLP learns
  non-linear decision boundaries. Most powerful model but needs labels.

ENSEMBLE VOTING:
  Each model casts a risk score 0.0–1.0.
  Final = 0.4×IsoForest + 0.25×DBSCAN + 0.35×MLP
  (MLP highest weight once trained on labelled data)
──────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib, os


FEATURE_COLS = [
    "invoice_amount",
    "po_amount",
    "amount_variance_pct",
    "quantity",
    "unit_price",
    "line_vs_invoice_pct",
    "days_to_due",
    "is_month_end",
    "is_friday",
    "vendor_risk_base",
    "processing_hour",
]


class EnsembleAnomalyDetector:
    """
    AI Layers 3+4+5: Three-model ensemble for numeric anomaly detection.
    Outputs ensemble risk score and individual model votes.
    """

    ENSEMBLE_WEIGHTS = {
        "isolation_forest": 0.40,
        "dbscan":           0.25,
        "mlp_nn":           0.35,
    }

    def __init__(self):
        self.scaler = StandardScaler()

        # Layer 3: Isolation Forest
        self.iso_forest = IsolationForest(
            n_estimators=300,
            contamination=0.08,     # 8% anomaly rate in ABC data
            max_samples="auto",
            random_state=42,
        )

        # Layer 4: DBSCAN (for clustering-based outlier detection)
        self.dbscan = DBSCAN(
            eps=1.5,                # Neighborhood radius (in scaled space)
            min_samples=5,          # Min points to form a cluster
            metric="euclidean",
        )

        # Layer 5: MLP Neural Network
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),   # 3-layer network
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
        )

        self.dbscan_labels_: np.ndarray = None
        self.dbscan_X_train_: np.ndarray = None
        self.is_fitted = False

    def _extract(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and scale features."""
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        for c in missing:
            df[c] = 0
        return df[FEATURE_COLS].fillna(0).values

    def fit(self, df: pd.DataFrame) -> "EnsembleAnomalyDetector":
        X_raw = self._extract(df)
        X     = self.scaler.fit_transform(X_raw)

        # Layer 3: Isolation Forest
        self.iso_forest.fit(X)

        # Layer 4: DBSCAN (train on normal data only)
        normal_mask = df["is_anomaly"].values == 0 if "is_anomaly" in df.columns else np.ones(len(X), dtype=bool)
        X_normal    = X[normal_mask]
        self.dbscan.fit(X_normal)
        self.dbscan_X_train_ = X_normal
        self.dbscan_labels_  = self.dbscan.labels_

        # Layer 5: MLP (needs labelled data)
        if "is_anomaly" in df.columns and df["is_anomaly"].sum() >= 5:
            y = df["is_anomaly"].values
            self.mlp.fit(X, y)
        else:
            # Fallback: use Isolation Forest labels to bootstrap MLP
            iso_labels = (self.iso_forest.predict(X) == -1).astype(int)
            self.mlp.fit(X, iso_labels)

        self.is_fitted = True
        return self

    def _iso_risk(self, x: np.ndarray) -> float:
        """Isolation Forest: convert decision function to 0–1 risk."""
        score = float(self.iso_forest.decision_function(x.reshape(1, -1))[0])
        # decision_function: positive=normal, negative=anomaly
        # Map to risk: score=-0.5 → risk≈1.0, score=0.5 → risk≈0.0
        return float(np.clip(0.5 - score, 0.0, 1.0))

    def _dbscan_risk(self, x: np.ndarray) -> float:
        """
        DBSCAN: measure how far this point is from ANY training cluster.
        Uses KNN distance to nearest training point.
        """
        if self.dbscan_X_train_ is None:
            return 0.5
        # Distance to 5 nearest training points
        diffs   = self.dbscan_X_train_ - x
        dists   = np.linalg.norm(diffs, axis=1)
        nearest = np.sort(dists)[:5]
        avg_dist = float(np.mean(nearest))
        # Normalize: dist=0 → risk=0.0, dist=3+ → risk=1.0
        return float(np.clip(avg_dist / 3.0, 0.0, 1.0))

    def _mlp_risk(self, x: np.ndarray) -> float:
        """MLP Neural Network: probability of anomaly class."""
        proba = self.mlp.predict_proba(x.reshape(1, -1))[0]
        # proba[1] = P(anomaly)
        return float(proba[1]) if len(proba) > 1 else 0.0

    def score(self, invoice: dict) -> dict:
        """
        Run all 3 models and return ensemble result.
        """
        row    = pd.DataFrame([invoice])
        X_raw  = self._extract(row)
        X      = self.scaler.transform(X_raw)[0]

        iso_risk   = self._iso_risk(X)
        dbscan_risk = self._dbscan_risk(X)
        mlp_risk   = self._mlp_risk(X)

        w = self.ENSEMBLE_WEIGHTS
        ensemble = (
            iso_risk    * w["isolation_forest"] +
            dbscan_risk * w["dbscan"]           +
            mlp_risk    * w["mlp_nn"]
        )
        ensemble = round(float(ensemble), 4)

        # Dominant model
        votes = {"IsolationForest": iso_risk, "DBSCAN": dbscan_risk, "MLP_NN": mlp_risk}
        dominant = max(votes, key=votes.get)

        return {
            "ensemble_risk":  ensemble,
            "model_votes": {
                "isolation_forest": round(iso_risk, 4),
                "dbscan":           round(dbscan_risk, 4),
                "mlp_nn":           round(mlp_risk, 4),
            },
            "dominant_model": dominant,
        }

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(self.scaler,      os.path.join(dir_path, "scaler.pkl"))
        joblib.dump(self.iso_forest,  os.path.join(dir_path, "iso_forest.pkl"))
        joblib.dump(self.mlp,         os.path.join(dir_path, "mlp.pkl"))
        joblib.dump(self.dbscan_X_train_, os.path.join(dir_path, "dbscan_train.pkl"))

    def load(self, dir_path: str) -> "EnsembleAnomalyDetector":
        self.scaler        = joblib.load(os.path.join(dir_path, "scaler.pkl"))
        self.iso_forest    = joblib.load(os.path.join(dir_path, "iso_forest.pkl"))
        self.mlp           = joblib.load(os.path.join(dir_path, "mlp.pkl"))
        self.dbscan_X_train_ = joblib.load(os.path.join(dir_path, "dbscan_train.pkl"))
        self.is_fitted = True
        return self
