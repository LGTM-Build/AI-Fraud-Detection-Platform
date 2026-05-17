# services/normalizer.py
# ─────────────────────────────────────────────────────────────
# Normalisasi record dari Node.js → format internal Python.
#
# Input  : module (procurement / expense) + raw dict dari Node.js
# Output : dict standar yang siap dipakai anomaly_detection_v2
#
# Field wajib output (harus sesuai FEATURE_COLS di model):
#   amountTotal, unitPrice, quantity,
#   purchaseDate, approvalDate, invoiceDate, paymentDate,
#   vendorRegistrationDate, contractDate,
#   vendorName, department, location, category,
#   itemDescription, transaction_type, status,
#   employeeId, purchaseId / expenseId
# ─────────────────────────────────────────────────────────────

import logging
from datetime import datetime
from typing   import Any

logger = logging.getLogger("fradara.normalizer")


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_str(val: Any, default: str = "") -> str:
    if val is None:
        return default
    return str(val).strip()


def _parse_date(val: Any) -> str | None:
    """Kembalikan string tanggal ISO (YYYY-MM-DD) atau None."""
    if not val:
        return None
    s = str(val).strip()
    # Coba beberapa format umum
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f",
                "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(s[:len(fmt)], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Fallback: ambil 10 karakter pertama kalau sudah ISO-like
    if len(s) >= 10 and s[4] == "-":
        return s[:10]
    return None


def _days_between(date_a: str | None, date_b: str | None) -> float:
    """Hitung selisih hari antara dua tanggal. Return 0 jika tidak valid."""
    if not date_a or not date_b:
        return 0.0
    try:
        a = datetime.strptime(date_a[:10], "%Y-%m-%d")
        b = datetime.strptime(date_b[:10], "%Y-%m-%d")
        return max(float((b - a).days), 0.0)
    except ValueError:
        return 0.0


def _date_features(date_str: str | None) -> dict:
    """Ekstrak fitur kalender dari tanggal."""
    if not date_str:
        return {
            "purchase_month"     : 1,
            "purchase_dayofweek" : 0,
            "purchase_quarter"   : 1,
            "is_weekend"         : 0,
        }
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return {
            "purchase_month"     : d.month,
            "purchase_dayofweek" : d.weekday(),       # 0=Senin, 6=Minggu
            "purchase_quarter"   : (d.month - 1) // 3 + 1,
            "is_weekend"         : int(d.weekday() >= 5),
        }
    except ValueError:
        return {
            "purchase_month"     : 1,
            "purchase_dayofweek" : 0,
            "purchase_quarter"   : 1,
            "is_weekend"         : 0,
        }


# Encoding kategorikal sederhana — konsisten dengan training
# (model di-train dengan label encoding, jadi kita pakai mapping yg sama)
_TRANSACTION_TYPE_MAP = {
    "procurement": 0, "expense": 1, "reimbursement": 2,
    "advance": 3, "petty cash": 4,
}
_DEPARTMENT_MAP = {
    "finance": 0, "it": 1, "hr": 2, "operations": 3, "marketing": 4,
    "sales": 5, "legal": 6, "procurement": 7, "general affairs": 8,
}
_LOCATION_MAP = {
    "jakarta": 0, "surabaya": 1, "bandung": 2, "medan": 3,
    "semarang": 4, "makassar": 5, "other": 6,
}
_CATEGORY_MAP = {
    "office supplies": 0, "it equipment": 1, "travel": 2,
    "meals": 3, "maintenance": 4, "marketing": 5,
    "professional services": 6, "utilities": 7, "other": 8,
}
_STATUS_MAP = {
    "approved": 0, "pending": 1, "rejected": 2,
    "cancelled": 3, "paid": 4,
}


def _encode(val: Any, mapping: dict, default: int = 0) -> int:
    key = _safe_str(val).lower()
    return mapping.get(key, default)


# ─────────────────────────────────────────────────────────────
# NORMALISASI PROCUREMENT
# ─────────────────────────────────────────────────────────────

def _normalize_procurement(raw: dict) -> dict:
    """
    Ubah record procurement dari Node.js ke format internal.
    Field output sesuai kolom dataset training.
    """
    amount_total  = _safe_float(raw.get("amountTotal"))
    unit_price    = _safe_float(raw.get("unitPrice"))
    quantity      = _safe_float(raw.get("quantity"), default=1.0)

    # Jika unitPrice tidak ada tapi amountTotal & quantity ada
    if unit_price == 0.0 and quantity > 0 and amount_total > 0:
        unit_price = amount_total / quantity

    purchase_date  = _parse_date(raw.get("purchaseDate"))
    approval_date  = _parse_date(raw.get("approvalDate"))
    invoice_date   = _parse_date(raw.get("invoiceDate"))
    payment_date   = _parse_date(raw.get("paymentDate"))
    vendor_reg_date= _parse_date(raw.get("vendorRegistrationDate"))
    contract_date  = _parse_date(raw.get("contractDate"))

    date_feats = _date_features(purchase_date)

    return {
        # Identitas
        "sourceId"               : _safe_str(raw.get("id")),
        "purchaseId"             : _safe_str(raw.get("purchaseId")),
        "employeeId"             : _safe_str(
            raw.get("employeeExternalRef") or raw.get("employeeId")
        ),
        "module"                 : "procurement",

        # Numerik utama
        "amountTotal"            : amount_total,
        "unitPrice"              : unit_price,
        "quantity"               : quantity,

        # Tanggal raw (string)
        "purchaseDate"           : purchase_date,
        "approvalDate"           : approval_date,
        "invoiceDate"            : invoice_date,
        "paymentDate"            : payment_date,
        "vendorRegistrationDate" : vendor_reg_date,
        "contractDate"           : contract_date,

        # Derived time features
        "days_to_approval"           : _days_between(purchase_date, approval_date),
        "days_purchase_to_invoice"   : _days_between(purchase_date, invoice_date),
        "days_invoice_to_payment"    : _days_between(invoice_date, payment_date),
        "vendor_age_at_contract"     : _days_between(vendor_reg_date, contract_date),

        # Kalender
        **date_feats,

        # Kategorikal raw
        "transaction_type"       : _safe_str(
            raw.get("transactionType") or raw.get("procurementMethod"),
            "procurement",
        ),
        "department"             : _safe_str(raw.get("department")),
        "location"               : _safe_str(raw.get("location")),
        "category"               : _safe_str(
            raw.get("category") or raw.get("procurementMethod")
        ),
        "itemDescription"        : _safe_str(raw.get("itemDescription")),
        "status"                 : _safe_str(raw.get("status"), "Pending"),
        "vendorName"             : _safe_str(raw.get("vendorName")),

        # Encoded (untuk model)
        "transaction_type_enc"   : _encode(
            raw.get("transactionType") or raw.get("procurementMethod"),
            _TRANSACTION_TYPE_MAP,
        ),
        "department_enc"         : _encode(raw.get("department"), _DEPARTMENT_MAP),
        "location_enc"           : _encode(raw.get("location"), _LOCATION_MAP),
        "category_enc"           : _encode(
            raw.get("category") or raw.get("procurementMethod"),
            _CATEGORY_MAP,
        ),
        "itemDescription_enc"    : abs(hash(_safe_str(raw.get("itemDescription")))) % 100,
        "status_enc"             : _encode(raw.get("status"), _STATUS_MAP),

        # Placeholder fitur statistik (akan diisi oleh anomaly_detection_v2
        # menggunakan cache histori company)
        "log_amount"             : 0.0,
        "log_unitprice"          : 0.0,
        "amount_zscore"          : 0.0,
        "amount_vs_category_median": 1.0,
        "amount_consistency_ratio" : 1.0,
        "amount_vs_employee_avg"   : 1.0,
        "employee_daily_tx"        : 1.0,
        "vendor_tx_count"          : 1.0,
        "invoice_frequency"        : 1.0,
        "employee_tx_frequency"    : 1.0,
    }


# ─────────────────────────────────────────────────────────────
# NORMALISASI EXPENSE
# ─────────────────────────────────────────────────────────────

def _normalize_expense(raw: dict) -> dict:
    """
    Ubah record expense dari Node.js ke format internal.
    Di-map ke field yang sama dengan procurement agar bisa
    dianalisis dengan model yang sama.
    """
    amount_total = _safe_float(raw.get("amountTotal"))
    unit_price   = _safe_float(raw.get("unitPrice"))
    quantity     = _safe_float(raw.get("quantity"), default=1.0)

    if unit_price == 0.0 and quantity > 0 and amount_total > 0:
        unit_price = amount_total / quantity

    expense_date = _parse_date(raw.get("expenseDate") or raw.get("transactionDate"))
    approval_date= _parse_date(raw.get("approvalDate"))
    payment_date = _parse_date(raw.get("paymentDate"))

    date_feats = _date_features(expense_date)

    merchant = _safe_str(raw.get("merchant"))
    description = _safe_str(raw.get("description") or raw.get("itemDescription"))

    return {
        # Identitas
        "sourceId"               : _safe_str(raw.get("id")),
        "purchaseId"             : _safe_str(raw.get("expenseId")),
        "expenseId"              : _safe_str(raw.get("expenseId")),
        "employeeId"             : _safe_str(
            raw.get("employeeExternalRef") or raw.get("employeeId")
        ),
        "module"                 : "expense",

        # Numerik utama
        "amountTotal"            : amount_total,
        "unitPrice"              : unit_price,
        "quantity"               : quantity,

        # Tanggal raw
        "purchaseDate"           : expense_date,
        "approvalDate"           : approval_date,
        "invoiceDate"            : None,
        "paymentDate"            : payment_date,
        "vendorRegistrationDate" : None,
        "contractDate"           : None,

        # Derived time features
        "days_to_approval"           : _days_between(expense_date, approval_date),
        "days_purchase_to_invoice"   : 0.0,
        "days_invoice_to_payment"    : _days_between(expense_date, payment_date),
        "vendor_age_at_contract"     : 0.0,

        # Kalender
        **date_feats,

        # Kategorikal raw
        "transaction_type"       : "expense",
        "department"             : _safe_str(raw.get("department")),
        "location"               : _safe_str(raw.get("location")),
        "category"               : _safe_str(raw.get("category")),
        "itemDescription"        : description,
        "status"                 : _safe_str(raw.get("status"), "Pending"),
        "vendorName"             : merchant,   # merchant = vendor untuk expense

        # Encoded
        "transaction_type_enc"   : _encode("expense", _TRANSACTION_TYPE_MAP),
        "department_enc"         : _encode(raw.get("department"), _DEPARTMENT_MAP),
        "location_enc"           : _encode(raw.get("location"), _LOCATION_MAP),
        "category_enc"           : _encode(raw.get("category"), _CATEGORY_MAP),
        "itemDescription_enc"    : abs(hash(description)) % 100,
        "status_enc"             : _encode(raw.get("status"), _STATUS_MAP),

        # Placeholder statistik
        "log_amount"             : 0.0,
        "log_unitprice"          : 0.0,
        "amount_zscore"          : 0.0,
        "amount_vs_category_median": 1.0,
        "amount_consistency_ratio" : 1.0,
        "amount_vs_employee_avg"   : 1.0,
        "employee_daily_tx"        : 1.0,
        "vendor_tx_count"          : 1.0,
        "invoice_frequency"        : 1.0,
        "employee_tx_frequency"    : 1.0,
    }


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def normalize(module: str, raw: dict) -> dict:
    """
    Entry point utama — dipanggil dari analysis_worker.py.

    Parameters:
        module : "procurement" atau "expense"
        raw    : dict transaksi mentah dari Node.js

    Returns:
        dict standar siap dipakai anomaly_detection_v2.analyze()
    """
    if module == "procurement":
        tx = _normalize_procurement(raw)
    elif module == "expense":
        tx = _normalize_expense(raw)
    else:
        logger.warning("Module tidak dikenal: %s — fallback ke procurement", module)
        tx = _normalize_procurement(raw)

    logger.debug(
        "Normalized [%s] id=%s amount=%.2f dept=%s",
        module,
        tx.get("purchaseId", "?"),
        tx.get("amountTotal", 0),
        tx.get("department", "-"),
    )
    return tx
