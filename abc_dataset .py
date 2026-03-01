"""
abc_dataset.py
──────────────────────────────────────────────────────────────────
ABC Company — Rich Training Dataset
Covers all 9 AI layers with realistic UAE manufacturing data
──────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random, hashlib

# ─────────────────────────────────────────────────
# MASTER DATA
# ─────────────────────────────────────────────────
VENDORS = {
    "V001": {"name":"Al Futtaim Steel Supplies",   "category":"Raw Material",    "range":(5000,80000),  "terms":45, "gl":"4000100", "cc":"CC_MANUF_001", "risk_base":0.05},
    "V002": {"name":"Emirates Logistics Ltd",       "category":"Logistics",       "range":(2000,25000),  "terms":30, "gl":"4000200", "cc":"CC_OPS_002",   "risk_base":0.07},
    "V003": {"name":"Gulf Packaging Co.",           "category":"Packaging",       "range":(1500,15000),  "terms":30, "gl":"4000300", "cc":"CC_MANUF_001", "risk_base":0.10},
    "V004": {"name":"TechParts MENA",               "category":"Spare Parts",     "range":(3000,50000),  "terms":60, "gl":"4000400", "cc":"CC_MAINT_003", "risk_base":0.04},
    "V005": {"name":"Arabia Office Supplies",       "category":"General Expenses","range":(200,5000),    "terms":15, "gl":"4000500", "cc":"CC_ADMIN_004", "risk_base":0.03},
}

ANOMALY_TYPES = [
    "overpayment",        # invoice 2-5× PO
    "ghost_vendor",       # unknown vendor_id
    "duplicate_fraud",    # same PO re-submitted
    "line_mismatch",      # qty×price ≠ invoice total
    "backdated",          # invoice date far in past
    "round_number",       # suspiciously round amounts
    "split_invoice",      # large PO split into small invoices to avoid threshold
]

EMAIL_TEMPLATES = {
    "normal": [
        "Please find attached invoice {inv} for services rendered. Amount: AED {amt}. PO reference: {po}.",
        "Invoice {inv} attached for your records. Total: AED {amt}. Please process by {due}.",
        "Kindly process invoice {inv} for AED {amt} against PO {po}. Payment due in {terms} days.",
    ],
    "urgent": [
        "URGENT: Invoice {inv} requires immediate processing. Amount: AED {amt}. Due today!",
        "Please prioritize invoice {inv} – AED {amt} payment overdue. PO: {po}.",
    ],
    "suspicious": [
        "Invoice {inv} for consulting services. Amount: AED {amt}. No PO attached.",
        "Revised invoice {inv} – please discard previous. New amount: AED {amt}.",
        "Final demand: Invoice {inv} AED {amt} unpaid for 90 days. Legal action pending.",
    ]
}


def generate_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate rich ABC Company invoice dataset with 25+ features."""
    np.random.seed(seed)
    random.seed(seed)

    records = []
    base_date = datetime(2024, 1, 1)

    for i in range(n):
        v_id = random.choice(list(VENDORS.keys()))
        v    = VENDORS[v_id]
        low, high = v["range"]

        # Base invoice
        po_amt     = round(np.random.uniform(low, high), 2)
        inv_amt    = round(po_amt * np.random.uniform(0.97, 1.04), 2)
        qty        = random.randint(5, 200)
        unit_price = round(inv_amt / qty, 4)
        inv_date   = base_date + timedelta(days=i * 0.8 + random.randint(0, 5))
        due_date   = inv_date + timedelta(days=v["terms"])
        days_due   = (due_date - datetime.today()).days

        # Email text (NLP feature)
        tpl = random.choice(EMAIL_TEMPLATES["normal"])
        email_text = tpl.format(
            inv=f"ABC-{2024*100+i:07d}", amt=f"{inv_amt:,.2f}",
            po=f"PO-{random.randint(10000,99999)}", due=due_date.strftime("%d-%b-%Y"),
            terms=v["terms"]
        )

        rec = {
            # Identity
            "invoice_number":  f"ABC-{2024*100+i:07d}",
            "vendor_id":       v_id,
            "vendor_name":     v["name"],
            "po_number":       f"PO-{random.randint(10000,99999)}",
            "category":        v["category"],
            "gl_account":      v["gl"],
            "cost_center":     v["cc"],
            "currency":        "AED",

            # Amounts
            "po_amount":       po_amt,
            "invoice_amount":  inv_amt,
            "quantity":        qty,
            "unit_price":      unit_price,

            # Dates
            "invoice_date":    inv_date.strftime("%Y-%m-%d"),
            "days_to_due":     max(days_due, -5),
            "payment_terms":   v["terms"],
            "processing_hour": random.randint(8, 17),   # Business hours

            # Derived features (AI will learn these)
            "amount_variance_pct":  round((inv_amt - po_amt) / po_amt * 100, 2),
            "line_total":           round(qty * unit_price, 2),
            "line_vs_invoice_pct":  0.0,  # filled below
            "is_month_end":         1 if inv_date.day >= 25 else 0,
            "is_friday":            1 if inv_date.weekday() == 4 else 0,
            "vendor_risk_base":     v["risk_base"],

            # NLP
            "email_text":      email_text,
            "email_urgency":   0,   # 0=normal, 1=urgent, 2=suspicious
            "email_word_count": len(email_text.split()),

            # Labels
            "is_anomaly":      0,
            "anomaly_type":    "none",
        }
        rec["line_vs_invoice_pct"] = round(abs(rec["line_total"] - inv_amt) / inv_amt * 100, 2)
        records.append(rec)

    # ── Inject 8% anomalies (40 records) ──────────────────
    anomaly_indices = random.sample(range(n), int(n * 0.08))
    for idx in anomaly_indices:
        atype = random.choice(ANOMALY_TYPES)
        r = records[idx]

        if atype == "overpayment":
            r["invoice_amount"] = round(r["po_amount"] * random.uniform(1.8, 4.0), 2)
            r["amount_variance_pct"] = round((r["invoice_amount"] - r["po_amount"]) / r["po_amount"] * 100, 2)

        elif atype == "ghost_vendor":
            r["vendor_id"]   = "V999"
            r["vendor_name"] = "Unknown Consulting LLC"
            r["invoice_amount"] = round(random.uniform(30000, 150000), 2)

        elif atype == "duplicate_fraud":
            # Reuse an earlier PO number
            earlier = records[max(0, idx-10)]
            r["po_number"]      = earlier["po_number"]
            r["invoice_amount"] = earlier["invoice_amount"]

        elif atype == "line_mismatch":
            r["quantity"]   = r["quantity"] * 10
            # invoice_amount stays same → line_total now 10× different
            r["line_vs_invoice_pct"] = round(abs(r["quantity"] * r["unit_price"] - r["invoice_amount"]) / r["invoice_amount"] * 100, 2)

        elif atype == "backdated":
            r["days_to_due"] = random.randint(-180, -60)
            r["email_urgency"] = 2

        elif atype == "round_number":
            r["invoice_amount"] = float(random.choice([50000, 100000, 150000, 200000, 75000]))

        elif atype == "split_invoice":
            # Just below the 50,000 AED approval threshold
            r["invoice_amount"] = round(random.uniform(48000, 49900), 2)

        # Suspicious email
        tpl = random.choice(EMAIL_TEMPLATES["suspicious"])
        r["email_text"]    = tpl.format(inv=r["invoice_number"], amt=f"{r['invoice_amount']:,.2f}", po=r["po_number"])
        r["email_urgency"] = 2
        r["is_anomaly"]    = 1
        r["anomaly_type"]  = atype

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────
# LIVE INVOICES (submitted today for processing)
# ─────────────────────────────────────────────────
LIVE_INVOICES = [
    {   # Clean — AUTO_POST expected
        "invoice_number":"ABC-2026-L001","vendor_id":"V001","vendor_name":"Al Futtaim Steel Supplies",
        "po_number":"PO-77001","po_amount":35000.00,"invoice_amount":35700.00,"quantity":42,"unit_price":850.00,
        "gl_account":"4000100","cost_center":"CC_MANUF_001","invoice_date":"2026-03-01",
        "days_to_due":44,"payment_terms":45,"currency":"AED","category":"Raw Material",
        "amount_variance_pct":2.0,"line_total":35700.00,"line_vs_invoice_pct":0.0,
        "is_month_end":0,"is_friday":0,"vendor_risk_base":0.05,"processing_hour":10,
        "email_text":"Please find invoice ABC-2026-L001 for steel supplies. AED 35,700 against PO-77001.",
        "email_urgency":0,"email_word_count":18,
    },
    {   # 3× overpayment — HOLD expected
        "invoice_number":"ABC-2026-L002","vendor_id":"V002","vendor_name":"Emirates Logistics Ltd",
        "po_number":"PO-77002","po_amount":8000.00,"invoice_amount":24000.00,"quantity":1,"unit_price":24000.00,
        "gl_account":"4000200","cost_center":"CC_OPS_002","invoice_date":"2026-03-01",
        "days_to_due":28,"payment_terms":30,"currency":"AED","category":"Logistics",
        "amount_variance_pct":200.0,"line_total":24000.00,"line_vs_invoice_pct":0.0,
        "is_month_end":0,"is_friday":0,"vendor_risk_base":0.07,"processing_hour":15,
        "email_text":"Revised invoice ABC-2026-L002 – please discard previous. New amount: AED 24,000.",
        "email_urgency":2,"email_word_count":14,
    },
    {   # Duplicate of L001 — REJECTED_DUPLICATE expected
        "invoice_number":"ABC-2026-L001","vendor_id":"V001","vendor_name":"Al Futtaim Steel Supplies",
        "po_number":"PO-77001","po_amount":35000.00,"invoice_amount":35700.00,"quantity":42,"unit_price":850.00,
        "gl_account":"4000100","cost_center":"CC_MANUF_001","invoice_date":"2026-03-01",
        "days_to_due":44,"payment_terms":45,"currency":"AED","category":"Raw Material",
        "amount_variance_pct":2.0,"line_total":35700.00,"line_vs_invoice_pct":0.0,
        "is_month_end":0,"is_friday":0,"vendor_risk_base":0.05,"processing_hour":11,
        "email_text":"Please find invoice ABC-2026-L001 for steel supplies. AED 35,700 against PO-77001.",
        "email_urgency":0,"email_word_count":18,
    },
    {   # Backdated 90 days — HOLD expected
        "invoice_number":"ABC-2026-L003","vendor_id":"V003","vendor_name":"Gulf Packaging Co.",
        "po_number":"PO-77003","po_amount":9500.00,"invoice_amount":9800.00,"quantity":60,"unit_price":163.33,
        "gl_account":"4000300","cost_center":"CC_MANUF_001","invoice_date":"2025-12-01",
        "days_to_due":-90,"payment_terms":30,"currency":"AED","category":"Packaging",
        "amount_variance_pct":3.2,"line_total":9800.0,"line_vs_invoice_pct":0.1,
        "is_month_end":0,"is_friday":0,"vendor_risk_base":0.10,"processing_hour":16,
        "email_text":"Final demand: Invoice ABC-2026-L003 AED 9,800 unpaid for 90 days. Legal action pending.",
        "email_urgency":2,"email_word_count":18,
    },
    {   # Normal small invoice — AUTO_POST
        "invoice_number":"ABC-2026-L004","vendor_id":"V005","vendor_name":"Arabia Office Supplies",
        "po_number":"PO-77004","po_amount":1200.00,"invoice_amount":1195.00,"quantity":20,"unit_price":59.75,
        "gl_account":"4000500","cost_center":"CC_ADMIN_004","invoice_date":"2026-03-01",
        "days_to_due":13,"payment_terms":15,"currency":"AED","category":"General Expenses",
        "amount_variance_pct":-0.4,"line_total":1195.0,"line_vs_invoice_pct":0.0,
        "is_month_end":0,"is_friday":0,"vendor_risk_base":0.03,"processing_hour":9,
        "email_text":"Invoice ABC-2026-L004 attached for your records. Total: AED 1,195. Please process.",
        "email_urgency":0,"email_word_count":14,
    },
    {   # Borderline — REVIEW expected
        "invoice_number":"ABC-2026-L005","vendor_id":"V004","vendor_name":"TechParts MENA",
        "po_number":"PO-77005","po_amount":20000.00,"invoice_amount":23500.00,"quantity":20,"unit_price":1175.00,
        "gl_account":"4000400","cost_center":"CC_MAINT_003","invoice_date":"2026-03-01",
        "days_to_due":57,"payment_terms":60,"currency":"AED","category":"Spare Parts",
        "amount_variance_pct":17.5,"line_total":23500.0,"line_vs_invoice_pct":0.0,
        "is_month_end":0,"is_friday":0,"vendor_risk_base":0.04,"processing_hour":14,
        "email_text":"Kindly process invoice ABC-2026-L005 for AED 23,500 against PO-77005. Payment due in 57 days.",
        "email_urgency":0,"email_word_count":17,
    },
]
