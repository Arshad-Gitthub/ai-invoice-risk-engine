"""
layer9_continuous_learning.py
──────────────────────────────────────────────────────────────
AI LAYER 9: Continuous Learning & Model Drift Monitor
──────────────────────────────────────────────────────────────
PURPOSE: The AI LEARNS from finance team feedback — when they
         approve or reject an invoice flagged by the system,
         that feedback updates the model weights.

         Also monitors for model drift: if the anomaly rate
         suddenly changes, it triggers retraining alerts.

AI TYPE: Online gradient descent weight update + statistical drift detection
  - PSI (Population Stability Index) for drift detection
  - Exponential weighted moving average for accuracy tracking
  - Automatic retraining trigger when drift > threshold
──────────────────────────────────────────────────────────────
"""

import numpy as np
from collections import deque
from datetime import datetime
import json


class ContinuousLearner:
    """
    AI Layer 9: Feedback-driven continuous improvement.
    """

    DRIFT_THRESHOLD   = 0.2     # PSI > 0.2 triggers retraining alert
    MIN_FEEDBACK      = 20      # Minimum feedback before weight updates

    def __init__(self):
        # Layer weight adjustments (starts neutral, adapts from feedback)
        self.layer_weights = {
            "nlp_risk":            0.10,
            "dedup_risk":          0.15,
            "iso_forest_risk":     0.18,
            "dbscan_risk":         0.12,
            "mlp_risk":            0.15,
            "vendor_profile_risk": 0.18,
            "po_tolerance_risk":   0.12,
        }

        self.feedback_log:     list[dict]  = []
        self.recent_scores:    deque       = deque(maxlen=100)
        self.accuracy_history: list[float] = []

        # Exponential weighted moving avg of accuracy
        self.ewma_accuracy: float = 0.0
        self.alpha: float = 0.1   # EWMA smoothing factor

    def record_feedback(self, invoice_number: str, system_decision: str,
                        human_decision: str, signals: dict) -> dict:
        """
        Record finance team feedback.
        system_decision: what AI decided (AUTO_POST/REVIEW/HOLD)
        human_decision:  what finance team confirmed (APPROVED/REJECTED)
        """
        correct = (
            (system_decision == "HOLD"      and human_decision == "REJECTED") or
            (system_decision == "AUTO_POST" and human_decision == "APPROVED") or
            (system_decision == "REVIEW"    and human_decision in ("APPROVED","REJECTED"))
        )

        # EWMA accuracy update
        self.ewma_accuracy = (self.alpha * float(correct)) + (1 - self.alpha) * self.ewma_accuracy

        # Log feedback
        entry = {
            "invoice_number":  invoice_number,
            "system_decision": system_decision,
            "human_decision":  human_decision,
            "correct":         correct,
            "timestamp":       datetime.utcnow().isoformat(),
            "signals":         signals,
        }
        self.feedback_log.append(entry)

        # Update layer weights if enough feedback
        weight_update = {}
        if len(self.feedback_log) >= self.MIN_FEEDBACK:
            weight_update = self._update_layer_weights()

        return {
            "recorded":       True,
            "correct":        correct,
            "ewma_accuracy":  round(self.ewma_accuracy, 4),
            "feedback_count": len(self.feedback_log),
            "weight_update":  weight_update,
        }

    def _update_layer_weights(self) -> dict:
        """
        Gradient-descent-style weight update:
        Increase weight of signals that correctly predicted the outcome.
        Decrease weight of signals that were high when they shouldn't be.
        """
        # Analyze recent 50 feedback entries
        recent = self.feedback_log[-50:]
        updates = {}

        for layer in self.layer_weights:
            correct_high = 0    # Signal high AND outcome was anomaly
            wrong_high   = 0    # Signal high AND outcome was normal
            total_high   = 0

            for entry in recent:
                sig_val = entry["signals"].get(layer, 0)
                if sig_val > 0.5:   # "high" signal
                    total_high += 1
                    if entry["human_decision"] == "REJECTED":
                        correct_high += 1
                    else:
                        wrong_high += 1

            if total_high > 5:
                precision = correct_high / total_high
                # Update: good precision → increase weight, poor → decrease
                delta = (precision - 0.5) * 0.02
                new_w = float(np.clip(self.layer_weights[layer] + delta, 0.02, 0.40))
                if abs(new_w - self.layer_weights[layer]) > 0.001:
                    updates[layer] = round(new_w, 4)
                    self.layer_weights[layer] = new_w

        # Renormalize weights to sum to 1.0
        total = sum(self.layer_weights.values())
        self.layer_weights = {k: round(v/total, 4) for k, v in self.layer_weights.items()}

        return updates

    def check_drift(self, new_scores: list[float]) -> dict:
        """
        PSI (Population Stability Index) drift detection.
        Compare current invoice risk score distribution to baseline.
        PSI > 0.2 = significant drift → retrain alert.
        """
        self.recent_scores.extend(new_scores)

        if len(self.recent_scores) < 20:
            return {"drift_detected": False, "psi": 0.0, "status": "Insufficient data"}

        # Compare historical vs recent distribution
        hist_scores  = list(self.recent_scores)[:len(self.recent_scores)//2]
        recent_sorted = list(self.recent_scores)[len(self.recent_scores)//2:]

        bins = np.linspace(0, 1, 11)
        hist_hist, _   = np.histogram(hist_scores,   bins=bins, density=True)
        recent_hist, _ = np.histogram(recent_sorted, bins=bins, density=True)

        # Add epsilon to avoid log(0)
        hist_hist   = hist_hist   + 1e-6
        recent_hist = recent_hist + 1e-6

        psi = float(np.sum((recent_hist - hist_hist) * np.log(recent_hist / hist_hist)))

        drift = psi > self.DRIFT_THRESHOLD
        return {
            "drift_detected":  drift,
            "psi":             round(psi, 4),
            "status":          "RETRAIN RECOMMENDED" if drift else "Stable",
            "ewma_accuracy":   round(self.ewma_accuracy, 4),
            "feedback_count":  len(self.feedback_log),
        }

    def get_model_health(self) -> dict:
        """Summary of AI system health for dashboard."""
        recent_50 = self.feedback_log[-50:] if self.feedback_log else []
        accuracy  = sum(1 for e in recent_50 if e["correct"]) / max(len(recent_50), 1)

        return {
            "total_feedback":     len(self.feedback_log),
            "recent_accuracy":    round(accuracy, 4),
            "ewma_accuracy":      round(self.ewma_accuracy, 4),
            "current_weights":    self.layer_weights,
            "status":             "HEALTHY" if accuracy > 0.85 else "DEGRADED",
        }
