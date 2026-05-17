# services/anomaly_detection_v2.py
# ─────────────────────────────────────────────────────────────
# Inference engine Fradara — Isolation Forest + Autoencoder Ensemble
#
# Alur per transaksi:
#   1. Hitung fitur statistik dari cache histori company
#   2. Susun feature vector sesuai FEATURE_COLS
#   3. Jalankan Isolation Forest (StandardScaler)
#   4. Jalankan Autoencoder (MinMaxScaler)
#   5. Weighted ensemble → fraud score 0–100
#   6. Tentukan riskLevel, intent, reasons
#   7. Update cache company
#   8. Return TransactionResult
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import logging
import math
import os
from functools import lru_cache
from pathlib   import Path
from typing    import Any

import joblib
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

logger = logging.getLogger("fradara.anomaly_detection")

# ─────────────────────────────────────────────────────────────
# LOAD CONFIG & MODELS  (singleton — load sekali saja)
# ─────────────────────────────────────────────────────────────

_MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))


@lru_cache(maxsize=1)
def _load_artifacts() -> dict:
    """Load semua model artifact. Dipanggil sekali, di-cache selamanya."""
    cfg_path = _MODELS_DIR / "threshold_config.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    iso        = joblib.load(_MODELS_DIR / "model_isolation_forest.pkl")
    std_scaler = joblib.load(_MODELS_DIR / "scaler_standard.pkl")
    mm_scaler  = joblib.load(_MODELS_DIR / "scaler_minmax.pkl")

    # Import TensorFlow hanya saat dibutuhkan (lazy) supaya startup lebih cepat
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    ae = tf.keras.models.load_model(_MODELS_DIR / "model_autoencoder.keras")

    logger.info(
        "Model artifacts loaded — threshold=%.4f W_ISO=%.1f W_AE=%.1f",
        cfg["threshold"], cfg["weightsISO"], cfg["weightsAE"],
    )
    return {
        "cfg"       : cfg,
        "iso"       : iso,
        "std_scaler": std_scaler,
        "mm_scaler" : mm_scaler,
        "ae"        : ae,
    }


# ─────────────────────────────────────────────────────────────
# FEATURE COLS — harus sama persis dengan training
# ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "amountTotal", "unitPrice", "quantity",
    "log_amount", "log_unitprice",
    "amount_zscore", "amount_vs_category_median",
    "amount_consistency_ratio", "amount_vs_employee_avg",
    "purchase_month", "purchase_dayofweek", "purchase_quarter", "is_weekend",
    "days_to_approval", "days_purchase_to_invoice",
    "days_invoice_to_payment", "vendor_age_at_contract",
    "employee_daily_tx", "vendor_tx_count",
    "invoice_frequency", "employee_tx_frequency",
    "transaction_type_enc", "department_enc", "location_enc",
    "category_enc", "itemDescription_enc", "status_enc",
]


# ─────────────────────────────────────────────────────────────
# FITUR STATISTIK DARI CACHE
# ─────────────────────────────────────────────────────────────

def _compute_stat_features(tx: dict, company_id: str | None) -> dict:
    """
    Hitung fitur statistik (amount_zscore, vendor_tx_count, dst)
    menggunakan histori dari company cache.
    Jika company_id None atau cache kosong, pakai nilai default netral.
    """
    amount = float(tx.get("amountTotal", 0))
    vendor = tx.get("vendorName", "")
    module = tx.get("module", "procurement")

    # Default — tidak ada histori
    stats = {
        "log_amount"               : math.log1p(amount),
        "log_unitprice"            : math.log1p(float(tx.get("unitPrice", 0))),
        "amount_zscore"            : 0.0,
        "amount_vs_category_median": 1.0,
        "amount_consistency_ratio" : 1.0,
        "amount_vs_employee_avg"   : 1.0,
        "employee_daily_tx"        : 1.0,
        "vendor_tx_count"          : 1.0,
        "invoice_frequency"        : 1.0,
        "employee_tx_frequency"    : 1.0,
    }

    if not company_id:
        return stats

    try:
        from cache.company_cache import load_vendor_history, load_expense_history

        if module == "procurement" and vendor:
            hist = load_vendor_history(company_id, vendor)
            if hist and hist.get("totalTransactions", 0) > 1:
                amounts = [
                    t["amountTotal"]
                    for t in hist.get("transactions", [])
                    if t.get("amountTotal", 0) > 0
                ]
                if amounts:
                    mean_a = float(np.mean(amounts))
                    std_a  = float(np.std(amounts)) + 1e-9
                    median_a = float(np.median(amounts))

                    stats["amount_zscore"]             = (amount - mean_a) / std_a
                    stats["amount_vs_category_median"] = amount / (median_a + 1e-9)
                    stats["amount_consistency_ratio"]  = amount / (mean_a + 1e-9)
                    stats["amount_vs_employee_avg"]    = amount / (mean_a + 1e-9)
                    stats["vendor_tx_count"]           = float(hist["totalTransactions"])
                    stats["invoice_frequency"]         = float(
                        hist.get("transactions", [{}])[-1:][0].get("isFraud", False)
                        or 1.0
                    )

        elif module == "expense":
            emp_ref = tx.get("employeeId", "")
            if emp_ref:
                hist = load_expense_history(company_id, emp_ref)
                if hist and hist.get("totalTransactions", 0) > 1:
                    amounts = [
                        t["amountTotal"]
                        for t in hist.get("transactions", [])
                        if t.get("amountTotal", 0) > 0
                    ]
                    if amounts:
                        mean_a   = float(np.mean(amounts))
                        std_a    = float(np.std(amounts)) + 1e-9
                        median_a = float(np.median(amounts))

                        stats["amount_zscore"]             = (amount - mean_a) / std_a
                        stats["amount_vs_category_median"] = amount / (median_a + 1e-9)
                        stats["amount_consistency_ratio"]  = amount / (mean_a + 1e-9)
                        stats["amount_vs_employee_avg"]    = amount / (mean_a + 1e-9)
                        stats["employee_daily_tx"]         = float(hist["totalTransactions"])
                        stats["employee_tx_frequency"]     = float(hist["totalTransactions"])

    except Exception as e:
        logger.warning("Gagal hitung stat features dari cache: %s", e)

    return stats


# ─────────────────────────────────────────────────────────────
# RISK LABEL
# ─────────────────────────────────────────────────────────────

def _risk_label(score: float) -> str:
    if score >= 75: return "HIGH"
    if score >= 45: return "MEDIUM"
    if score >= 20: return "LOW"
    return "SAFE"


# ─────────────────────────────────────────────────────────────
# INTENT DETECTION
# ─────────────────────────────────────────────────────────────

def _detect_intent(tx: dict, iso_s: float, ae_s: float) -> tuple[str, float, str]:
    """
    Return (intent_type, confidence, description).
    Urutan pengecekan dari yang paling spesifik ke umum.
    """
    amount_zscore  = float(tx.get("amount_zscore", 0))
    vs_median      = float(tx.get("amount_vs_category_median", 1))
    vendor_tx      = float(tx.get("vendor_tx_count", 1))
    employee_daily = float(tx.get("employee_daily_tx", 1))
    invoice_freq   = float(tx.get("invoice_frequency", 1))
    is_weekend     = int(tx.get("is_weekend", 0))
    vendor_age     = float(tx.get("vendor_age_at_contract", 999))

    if vs_median > 5 or amount_zscore > 3:
        return (
            "INFLATED_PRICE",
            min(0.5 + (vs_median - 5) * 0.05, 0.99),
            f"Nominal {vs_median:.1f}× di atas median kategori (z-score: {amount_zscore:.2f})",
        )

    if vendor_tx < 2 and 0 < vendor_age < 30:
        return (
            "DUPLICATE_VENDOR",
            0.72,
            f"Vendor baru (<30 hari) dengan riwayat transaksi sangat sedikit ({int(vendor_tx)} tx)",
        )

    if employee_daily > 5:
        return (
            "SPLIT_TRANSACTION",
            min(0.5 + (employee_daily - 5) * 0.04, 0.97),
            f"Transaksi burst: {int(employee_daily)}× dalam satu hari oleh karyawan yang sama",
        )

    if invoice_freq > 1:
        return (
            "SPLIT_TRANSACTION",
            min(0.6 + invoice_freq * 0.05, 0.97),
            f"Invoice muncul {int(invoice_freq)}× — kemungkinan transaksi duplikat",
        )

    if is_weekend:
        return (
            "WEEKEND_TRANSACTION",
            0.60,
            "Transaksi dilakukan di akhir pekan — di luar hari kerja normal",
        )

    if ae_s > 0.65:
        return (
            "UNUSUAL_PATTERN",
            float(ae_s),
            f"Autoencoder mendeteksi pola tidak dikenal (reconstruction error tinggi: {ae_s*100:.1f})",
        )

    if iso_s > 0.65:
        return (
            "ABNORMAL_AMOUNT",
            float(iso_s),
            f"Isolation Forest mengisolasi transaksi ini dari distribusi normal (score: {iso_s*100:.1f})",
        )

    return (
        "NORMAL",
        1.0 - max(iso_s, ae_s),
        "Tidak ada pola mencurigakan terdeteksi",
    )


# ─────────────────────────────────────────────────────────────
# AUTO REASONS
# ─────────────────────────────────────────────────────────────

def _auto_reasons(tx: dict, iso_s: float, ae_s: float) -> list[str]:
    reasons = []

    if tx.get("days_to_approval", 0) < 0:
        reasons.append(
            f"Approval date mendahului purchase date ({int(tx.get('days_to_approval', 0))} hari)"
        )
    if tx.get("days_invoice_to_payment", 0) < 0:
        reasons.append(
            f"Payment date mendahului invoice date ({int(tx.get('days_invoice_to_payment', 0))} hari)"
        )
    if 0 < tx.get("vendor_age_at_contract", 999) < 30:
        reasons.append(
            f"Vendor sangat baru ({int(tx.get('vendor_age_at_contract', 0))} hari sejak registrasi)"
        )
    if tx.get("amount_zscore", 0) > 3:
        reasons.append(
            f"Nominal jauh di atas rata-rata (z-score: {tx.get('amount_zscore', 0):.2f})"
        )
    if tx.get("amount_vs_category_median", 0) > 5:
        reasons.append(
            f"Nominal {tx.get('amount_vs_category_median', 0):.1f}× di atas median kategori"
        )
    if tx.get("employee_daily_tx", 0) > 5:
        reasons.append(
            f"Transaksi burst: {int(tx.get('employee_daily_tx', 0))}× dalam satu hari"
        )
    if tx.get("invoice_frequency", 0) > 1:
        reasons.append(
            f"Invoice duplikat (muncul {int(tx.get('invoice_frequency', 0))}×)"
        )
    if tx.get("employee_tx_frequency", 1) <= 2:
        reasons.append("Employee ID tidak dikenali atau sangat jarang bertransaksi")
    if iso_s > 0.65:
        reasons.append(
            f"Isolation Forest: terisolasi dari distribusi normal (score: {iso_s*100:.1f})"
        )
    if ae_s > 0.65:
        reasons.append(
            f"Autoencoder: pola tidak dikenali, reconstruction error tinggi (score: {ae_s*100:.1f})"
        )

    return reasons if reasons else ["Tidak ada indikator spesifik terdeteksi"]


# ─────────────────────────────────────────────────────────────
# UPDATE CACHE SETELAH ANALISIS
# ─────────────────────────────────────────────────────────────

def _update_cache(tx: dict, company_id: str | None, is_fraud: bool) -> None:
    if not company_id:
        return
    try:
        from cache.company_cache import update_vendor_cache, update_expense_cache
        module = tx.get("module", "procurement")
        if module == "procurement":
            vendor = tx.get("vendorName", "")
            if vendor:
                update_vendor_cache(company_id, vendor, tx, is_fraud)
        else:
            emp_ref = tx.get("employeeId", "")
            dept    = tx.get("department")
            if emp_ref:
                update_expense_cache(company_id, emp_ref, dept, tx, is_fraud)
    except Exception as e:
        logger.warning("Gagal update cache: %s", e)


# ─────────────────────────────────────────────────────────────
# MAIN ANALYZE FUNCTION
# ─────────────────────────────────────────────────────────────

def analyze(job_id: str, tx: dict, company_id: str | None) -> Any:
    """
    Analisis satu transaksi dengan model ensemble.

    Parameters:
        job_id     : ID job (untuk logging)
        tx         : dict hasil normalize() dari normalizer.py
        company_id : scope cache perusahaan (boleh None)

    Returns:
        TransactionResult (Pydantic model)
    """
    from schemas import (
        TransactionResult, ModelScores,
        IntentResult, IntentType, RiskLevel, ModuleType,
    )

    arts = _load_artifacts()
    cfg  = arts["cfg"]

    # ── 1. Hitung fitur statistik dari cache ──────────────────
    stat_feats = _compute_stat_features(tx, company_id)
    tx.update(stat_feats)   # merge back ke tx dict

    # ── 2. Susun feature vector ───────────────────────────────
    feat_vec = np.array(
        [float(tx.get(col, 0.0)) for col in FEATURE_COLS],
        dtype=np.float32,
    ).reshape(1, -1)

    # ── 3. Isolation Forest score ─────────────────────────────
    X_std      = arts["std_scaler"].transform(feat_vec)
    iso_raw    = float(-arts["iso"].score_samples(X_std)[0])
    iso_norm   = float(
        np.clip(
            (iso_raw - cfg["isoMin"]) / (cfg["isoMax"] - cfg["isoMin"] + 1e-9),
            0, 1,
        )
    )

    # ── 4. Autoencoder reconstruction error ───────────────────
    X_mm       = arts["mm_scaler"].transform(feat_vec)
    recon      = arts["ae"].predict(X_mm, verbose=0)
    ae_raw     = float(np.mean((X_mm - recon) ** 2))
    ae_norm    = float(
        np.clip(
            (ae_raw - cfg["aeMin"]) / (cfg["aeMax"] - cfg["aeMin"] + 1e-9),
            0, 1,
        )
    )

    # ── 5. Weighted ensemble ──────────────────────────────────
    W_ISO    = cfg["weightsISO"]   # 0.30
    W_AE     = cfg["weightsAE"]    # 0.70
    ens_score = W_ISO * iso_norm + W_AE * ae_norm
    fraud_score_100 = round(float(ens_score * 100), 2)

    predicted_fraud = ens_score >= cfg["threshold"]
    risk            = _risk_label(fraud_score_100)

    # ── 6. Intent & reasons ───────────────────────────────────
    intent_type, intent_conf, intent_desc = _detect_intent(tx, iso_norm, ae_norm)
    reasons = _auto_reasons(tx, iso_norm, ae_norm)

    logger.info(
        "[%s] id=%s score=%.1f fraud=%s risk=%s intent=%s",
        job_id,
        tx.get("purchaseId", "?"),
        fraud_score_100,
        predicted_fraud,
        risk,
        intent_type,
    )

    # ── 7. Update cache ───────────────────────────────────────
    _update_cache(tx, company_id, bool(predicted_fraud))

    # ── 8. Susun result ───────────────────────────────────────
    module = tx.get("module", "procurement")

    return TransactionResult(
        transactionId   = tx.get("purchaseId", ""),
        module          = ModuleType(module),
        fraudScore      = fraud_score_100,
        predictedFraud  = bool(predicted_fraud),
        riskLevel       = RiskLevel(risk),
        scores          = ModelScores(
            isolationForest = round(iso_norm * 100, 2),
            autoencoder     = round(ae_norm  * 100, 2),
            ensemble        = round(ens_score * 100, 2),
            fraudScore      = fraud_score_100,
        ),
        intent          = IntentResult(
            intent      = IntentType(intent_type),
            confidence  = round(intent_conf, 4),
            description = intent_desc,
        ),
        reasons         = reasons,
        amountTotal     = tx.get("amountTotal"),
        department      = tx.get("department") or None,
        vendorName      = tx.get("vendorName") or None,
        merchant        = tx.get("vendorName") or None if module == "expense" else None,
        category        = tx.get("category") or None,
        transactionDate = tx.get("purchaseDate") or None,
    )