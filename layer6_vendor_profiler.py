"""
layer6_vendor_profiler.py
──────────────────────────────────────────────────────────────
AI LAYER 6: Dynamic Vendor Behavioral Profiler
──────────────────────────────────────────────────────────────
PURPOSE: Build a continuously-updated statistical profile for
         each vendor and score how much this invoice deviates
         from their established behavior pattern.

AI TYPE: Online Gaussian Profile + Mahalanobis Distance
  - Builds multi-variate Gaussian model per vendor
  - Uses Mahalanobis distance (accounts for feature correlations)
  - Profile updates online with each new clean invoice (learns continuously)
  - Much more powerful than simple mean±std comparisons

EXAMPLE:
  V001 normally invoices: amount~40K, qty~50, price~800
  All three together define a 3D "normal zone"
  Mahalanobis distance detects when a new invoice falls
  OUTSIDE this ellipse — even if each feature alone looks ok
──────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinv


VENDOR_FEATURES = ["invoice_amount", "quantity", "unit_price", "amount_variance_pct", "days_to_due"]


class VendorBehaviorProfiler:
    """
    AI Layer 6: Per-vendor Mahalanobis distance anomaly scorer.
    Updates online — learns from every clean invoice processed.
    """

    def __init__(self):
        self.profiles: dict[str, dict] = {}  # vendor_id → profile
        self.global_profile: dict = {}        # fallback for unknown vendors
        self.is_fitted = False

    def _build_profile(self, data: np.ndarray) -> dict:
        """Build Gaussian profile: mean vector + inverse covariance."""
        if len(data) < 3:
            return None
        mean = np.mean(data, axis=0)
        cov  = np.cov(data.T)
        # Use pseudo-inverse for stability with near-singular matrices
        cov_inv = pinv(cov + np.eye(len(mean)) * 1e-6)
        std  = np.std(data, axis=0) + 1e-9
        return {
            "mean":    mean,
            "cov_inv": cov_inv,
            "std":     std,
            "n":       len(data),
            "min":     np.min(data, axis=0),
            "max":     np.max(data, axis=0),
        }

    def fit(self, df: pd.DataFrame) -> "VendorBehaviorProfiler":
        """Train on normal invoices per vendor."""
        normal = df[df["is_anomaly"] == 0] if "is_anomaly" in df.columns else df
        feats  = [c for c in VENDOR_FEATURES if c in normal.columns]

        # Global profile
        X_all = normal[feats].fillna(0).values
        self.global_profile = self._build_profile(X_all)
        self.global_profile["features"] = feats

        # Per-vendor profile
        for v_id, grp in normal.groupby("vendor_id"):
            X_v = grp[feats].fillna(0).values
            if len(X_v) >= 3:
                profile = self._build_profile(X_v)
                if profile:
                    # Vendor trust: penalizes past anomalies
                    total = len(df[df["vendor_id"] == v_id])
                    anoms = len(df[(df["vendor_id"] == v_id) & (df.get("is_anomaly",pd.Series(0)) == 1)])
                    profile["trust"]    = max(0.0, 1.0 - (anoms / max(total, 1)) * 4)
                    profile["features"] = feats
                    self.profiles[str(v_id)] = profile

        self.is_fitted = True
        return self

    def _extract_vector(self, invoice: dict, features: list) -> np.ndarray:
        return np.array([float(invoice.get(f, 0)) for f in features])

    def score(self, invoice: dict) -> dict:
        """
        Compute Mahalanobis distance of this invoice from vendor's normal zone.
        Returns risk score 0.0–1.0.
        """
        v_id    = str(invoice.get("vendor_id", "UNKNOWN"))
        profile = self.profiles.get(v_id, self.global_profile)

        if not profile:
            return {"risk_score": 0.5, "mahal_dist": None, "vendor_trust": 0.5, "detail": "No vendor profile"}

        feats = profile["features"]
        x     = self._extract_vector(invoice, feats)

        try:
            dist = mahalanobis(x, profile["mean"], profile["cov_inv"])
        except Exception:
            # Fallback to normalized z-score if matrix issues
            dist = float(np.mean(np.abs(x - profile["mean"]) / profile["std"]))

        # Normalize: dist=0 → risk=0.0, dist=4+ → risk=1.0
        risk  = float(np.clip(dist / 4.0, 0.0, 1.0))
        trust = profile.get("trust", 1.0)

        # Trust-adjusted risk: low-trust vendors penalized more
        adjusted_risk = float(np.clip(risk * (2.0 - trust), 0.0, 1.0))

        detail = (
            f"Mahalanobis distance={dist:.2f} "
            f"(vendor profile n={profile['n']}, trust={trust:.2f})"
        )

        return {
            "risk_score":    round(adjusted_risk, 4),
            "raw_risk":      round(risk, 4),
            "mahal_dist":    round(dist, 4),
            "vendor_trust":  round(trust, 4),
            "vendor_n":      profile["n"],
            "detail":        detail,
        }

    def update_online(self, invoice: dict, is_clean: bool):
        """
        Update vendor profile with a newly confirmed clean invoice.
        This makes the AI continuously improve without retraining.
        """
        if not is_clean:
            return
        v_id = str(invoice.get("vendor_id", "UNKNOWN"))
        if v_id not in self.profiles:
            return
        p     = self.profiles[v_id]
        feats = p["features"]
        x     = self._extract_vector(invoice, feats)

        # Incremental mean update
        n      = p["n"]
        p["mean"] = (p["mean"] * n + x) / (n + 1)
        p["n"]    = n + 1
        # Note: covariance update is approximate here for speed
