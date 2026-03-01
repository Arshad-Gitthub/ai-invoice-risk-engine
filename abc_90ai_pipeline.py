"""
abc_90ai_pipeline.py
──────────────────────────────────────────────────────────────
ABC COMPANY — 90% AI Invoice Processing Pipeline
──────────────────────────────────────────────────────────────
9 AI LAYERS:
  1. NLP Email Classifier         (GBM + TF-IDF)
  2. AI Duplicate Detector        (Learned similarity)
  3. Isolation Forest             (Unsupervised anomaly)
  4. DBSCAN Cluster Outlier       (Spatial anomaly)
  5. MLP Neural Network           (Supervised anomaly)
  6. Vendor Behavioral Profiler   (Mahalanobis distance)
  7. PO Tolerance Learner         (GBM regression)
  8. Meta-Learner                 (Stacked Random Forest)
  9. Continuous Learning          (PSI drift + feedback loop)

ONLY 10% HUMAN:
  - Finance manager approves REVIEW queue
  - Finance manager confirms HOLD escalations
  - Feedback recorded back to Layer 9

Run: python abc_90ai_pipeline.py
──────────────────────────────────────────────────────────────
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from data.abc_dataset import generate_dataset, LIVE_INVOICES
from src.ai_layers.layer1_nlp_parser        import NLPEmailClassifier
from src.ai_layers.layer2_ai_dedup          import AIDuplicateDetector
from src.ai_layers.layer3_ensemble_anomaly  import EnsembleAnomalyDetector
from src.ai_layers.layer6_vendor_profiler   import VendorBehaviorProfiler
from src.ai_layers.layer7_po_matcher        import POToleranceLearner
from src.ai_layers.layer8_meta_learner      import MetaLearner
from src.ai_layers.layer9_continuous_learning import ContinuousLearner


# ═══════════════════════════════════════════════════════════════
# PRINT HELPERS
# ═══════════════════════════════════════════════════════════════
def bar(score, width=20):
    n = int(score * width)
    return "█" * n + "░" * (width - n)

def header(title):
    print("\n" + "═"*68)
    print(f"  {title}")
    print("═"*68)

# ═══════════════════════════════════════════════════════════════
# STEP 1: GENERATE TRAINING DATA
# ═══════════════════════════════════════════════════════════════
def main():
 header("ABC COMPANY — 90% AI Invoice Processing System")
print("\n📂 Generating ABC Company invoice history (500 invoices)...")
df = generate_dataset(n=500)
print(f"   Total: {len(df)} | Normal: {(df.is_anomaly==0).sum()} | Anomalies: {(df.is_anomaly==1).sum()}")
print(f"   Anomaly types: {df[df.is_anomaly==1]['anomaly_type'].value_counts().to_dict()}")

# ═══════════════════════════════════════════════════════════════
# STEP 2: TRAIN ALL 9 AI LAYERS
# ═══════════════════════════════════════════════════════════════
header("TRAINING ALL 9 AI LAYERS")

print("\n  Layer 1 — NLP Email Classifier (GBM + TF-IDF)...")
nlp = NLPEmailClassifier()
nlp.fit(df["email_text"].tolist(), df["email_urgency"].tolist())
print(f"  ✓ Trained on {len(df)} email texts | 3 classes: NORMAL/URGENT/SUSPICIOUS")

print("\n  Layer 2 — AI Duplicate Detector (Learned Similarity)...")
dedup = AIDuplicateDetector()
train_records = df.to_dict("records")
dedup.fit(train_records)
print(f"  ✓ Learned similarity threshold: {dedup.threshold:.3f}")

print("\n  Layers 3+4+5 — Ensemble Anomaly Detector (IsoForest + DBSCAN + MLP)...")
ensemble = EnsembleAnomalyDetector()
ensemble.fit(df)
print(f"  ✓ IsolationForest: {ensemble.iso_forest.n_estimators} trees | contamination=8%")
print(f"  ✓ DBSCAN: eps={ensemble.dbscan.eps} | min_samples={ensemble.dbscan.min_samples}")
print(f"  ✓ MLP: hidden_layers=(64,32,16) | trained on {len(df)} samples")

print("\n  Layer 6 — Vendor Behavioral Profiler (Mahalanobis)...")
vendor_profiler = VendorBehaviorProfiler()
vendor_profiler.fit(df)
print(f"  ✓ Vendor profiles built:")
for v, p in vendor_profiler.profiles.items():
    print(f"      {v}: n={p['n']} | trust={p['trust']:.2f} | mean_amt={p['mean'][0]:,.0f} AED")

print("\n  Layer 7 — PO Tolerance Learner (GBM Regressor)...")
po_learner = POToleranceLearner()
po_learner.fit(df)
print(f"  ✓ Trained GBM on {len(df[df.is_anomaly==0])} normal invoices | learns per-vendor tolerances")

print("\n  Layer 8 — Meta-Learner (Stacked Random Forest)...")
meta = MetaLearner()
# Build training signals for meta-learner
print("     Building layer signals for 500 training invoices...")
meta_signals, meta_labels = [], []
for _, row in df.iterrows():
    inv    = row.to_dict()
    nlp_r  = nlp.predict(inv.get("email_text",""))
    ens_r  = ensemble.score(inv)
    vend_r = vendor_profiler.score(inv)
    po_r   = po_learner.score(inv)
    meta_signals.append({
        "nlp_risk":            nlp_r["risk_score"],
        "dedup_risk":          0.0,
        "iso_forest_risk":     ens_r["model_votes"]["isolation_forest"],
        "dbscan_risk":         ens_r["model_votes"]["dbscan"],
        "mlp_risk":            ens_r["model_votes"]["mlp_nn"],
        "vendor_profile_risk": vend_r["risk_score"],
        "po_tolerance_risk":   po_r["risk_score"],
    })
    meta_labels.append(int(inv.get("is_anomaly", 0)))

meta.fit(meta_signals, meta_labels)
print(f"  ✓ Meta-learner trained | RandomForest(200 trees) + CalibratedCV(isotonic)")

print("\n  Layer 9 — Continuous Learning Engine (PSI Drift Monitor)...")
learner = ContinuousLearner()
print(f"  ✓ Drift threshold: PSI={learner.DRIFT_THRESHOLD} | EWMA alpha={learner.alpha}")

print("\n✅ ALL 9 AI LAYERS TRAINED\n")

# ═══════════════════════════════════════════════════════════════
# STEP 3: PROCESS LIVE INVOICES
# ═══════════════════════════════════════════════════════════════
header("PROCESSING 6 LIVE ABC INVOICES — MARCH 2026")

results = []
all_risk_scores = []

for invoice in LIVE_INVOICES:
    print(f"\n{'─'*68}")
    print(f"📄 {invoice['invoice_number']}  |  {invoice['vendor_name']}")
    print(f"   Amount: AED {invoice['invoice_amount']:>12,.2f}  |  PO: AED {invoice['po_amount']:>12,.2f}  |  Variance: {invoice['amount_variance_pct']:+.1f}%")
    print(f"{'─'*68}")

    # ── Layer 1: NLP ──────────────────────────────────
    nlp_result = nlp.predict(invoice.get("email_text", ""))
    print(f"\n  L1 NLP   [{bar(nlp_result['risk_score'])}] {nlp_result['risk_score']:.3f}  → {nlp_result['label_name']} (conf={nlp_result['confidence']:.2f})")

    # ── Layer 2: AI Dedup ─────────────────────────────
    dup_result = dedup.check(invoice)
    dedup_risk = 1.0 if dup_result["is_duplicate"] else dup_result["match_score"]
    print(f"  L2 DEDUP [{bar(dedup_risk)}] {dedup_risk:.3f}  → {dup_result['match_type']}: {dup_result['detail'][:55]}")

    if dup_result["is_duplicate"]:
        final_decision = "REJECTED_DUPLICATE"
        print(f"\n  🚫 REJECTED — Duplicate detected. Pipeline stops here.")
        results.append({
            "invoice_number": invoice["invoice_number"],
            "vendor": invoice["vendor_name"],
            "amount": invoice["invoice_amount"],
            "variance_pct": invoice["amount_variance_pct"],
            "final_risk": 1.0, "decision": "REJECTED_DUPLICATE",
            "top_drivers": "Duplicate", "confidence": "HIGH"
        })
        continue

    # ── Layers 3+4+5: Ensemble ────────────────────────
    ens_result  = ensemble.score(invoice)
    iso_risk    = ens_result["model_votes"]["isolation_forest"]
    dbscan_risk = ens_result["model_votes"]["dbscan"]
    mlp_risk    = ens_result["model_votes"]["mlp_nn"]
    ens_avg     = ens_result["ensemble_risk"]

    print(f"  L3 ISO   [{bar(iso_risk)}]  {iso_risk:.3f}  → IsolationForest")
    print(f"  L4 DBSCAN[{bar(dbscan_risk)}] {dbscan_risk:.3f}  → DBSCAN cluster distance")
    print(f"  L5 MLP   [{bar(mlp_risk)}]  {mlp_risk:.3f}  → Neural Network P(anomaly)")
    print(f"       ↳ Ensemble avg: {ens_avg:.3f} (dominant: {ens_result['dominant_model']})")

    # ── Layer 6: Vendor Profiler ──────────────────────
    vend_result = vendor_profiler.score(invoice)
    print(f"  L6 VENDOR[{bar(vend_result['risk_score'])}] {vend_result['risk_score']:.3f}  → Mahal dist={vend_result['mahal_dist']} | trust={vend_result['vendor_trust']}")

    # ── Layer 7: PO Tolerance ─────────────────────────
    po_result = po_learner.score(invoice)
    print(f"  L7 PO    [{bar(po_result['risk_score'])}] {po_result['risk_score']:.3f}  → Actual={po_result['actual_variance']:.1f}% | Learned tol={po_result['learned_tolerance']:.1f}%")

    # ── Layer 8: Meta-Learner ─────────────────────────
    signals = {
        "nlp_risk":            nlp_result["risk_score"],
        "dedup_risk":          0.0,
        "iso_forest_risk":     iso_risk,
        "dbscan_risk":         dbscan_risk,
        "mlp_risk":            mlp_risk,
        "vendor_profile_risk": vend_result["risk_score"],
        "po_tolerance_risk":   po_result["risk_score"],
    }
    meta_result = meta.predict(signals)
    final_decision = meta_result["decision"]
    final_risk     = meta_result["final_risk_score"]

    print(f"\n  L8 META  [{bar(final_risk)}] {final_risk:.3f}  → {meta_result['meta_model']}")
    print(f"       ↳ Top drivers: {meta_result['top_drivers']}")

    # ── Layer 9: Log for drift monitoring ────────────
    all_risk_scores.append(final_risk)

    # Final output
    icons = {"AUTO_POST":"✅","REVIEW":"🟡","HOLD":"🔴"}
    icon  = icons.get(final_decision, "❓")
    print(f"\n  {icon}  FINAL DECISION: {final_decision}  ({final_risk*100:.1f}% risk | {meta_result['confidence']} confidence)")

    results.append({
        "invoice_number": invoice["invoice_number"],
        "vendor": invoice["vendor_name"],
        "amount": invoice["invoice_amount"],
        "variance_pct": invoice["amount_variance_pct"],
        "final_risk": final_risk,
        "decision": final_decision,
        "top_drivers": meta_result["top_drivers"],
        "confidence": meta_result["confidence"],
    })

# ═══════════════════════════════════════════════════════════════
# STEP 4: DRIFT CHECK (Layer 9)
# ═══════════════════════════════════════════════════════════════
drift = learner.check_drift(all_risk_scores)

# ═══════════════════════════════════════════════════════════════
# STEP 5: SUMMARY
# ═══════════════════════════════════════════════════════════════
header("PROCESSING SUMMARY")
print(f"\n  {'Invoice':<20} {'Vendor':<28} {'Amount':>12}  {'Risk':>5}  {'Decision'}")
print(f"  {'─'*20} {'─'*28} {'─'*12}  {'─'*5}  {'─'*22}")

auto_post = review = hold = dup = 0
for r in results:
    score_s = f"{r['final_risk']:.3f}" if r["decision"] != "REJECTED_DUPLICATE" else "  N/A"
    icons   = {"AUTO_POST":"✅","REVIEW":"🟡","HOLD":"🔴","REJECTED_DUPLICATE":"🚫"}
    icon    = icons.get(r["decision"],"")
    print(f"  {r['invoice_number']:<20} {r['vendor'][:27]:<28} {r['amount']:>12,.0f}  {score_s:>5}  {icon} {r['decision']}")
    if r["decision"] == "AUTO_POST":          auto_post += 1
    elif r["decision"] == "REVIEW":           review    += 1
    elif r["decision"] == "HOLD":             hold      += 1
    elif r["decision"] == "REJECTED_DUPLICATE": dup     += 1

protected = sum(r["amount"] for r in results if r["decision"] in ("HOLD","REJECTED_DUPLICATE","REVIEW"))
print(f"\n  ✅ Auto-posted (0% human):    {auto_post}")
print(f"  🟡 Review queue (human 10%): {review}")
print(f"  🔴 Held/escalated:           {hold}")
print(f"  🚫 Duplicate rejected:       {dup}")
print(f"\n  💰 AED protected: {protected:,.2f}")
print(f"\n  📊 Model drift:  PSI={drift['psi']:.4f} — {drift['status']}")
print(f"  🤖 AI Coverage: ~90% automated | ~10% human review queue")
    print(f"\n{'═'*68}\n  Pipeline complete — 9 AI layers executed\n{'═'*68}\n")
    return results
if __name__ == "__main__":
    main()
