# cache/company_cache.py
# ─────────────────────────────────────────────────────────────
# Company-scoped JSON cache untuk histori vendor dan expense.
#
# Struktur folder:
#   tmp/
#   └── {companyId}/
#       ├── vendor_{slug}.json
#       └── expense_{slug}.json
#
# Cara kerja:
#   1. Request masuk dengan companyId
#   2. Cek cache → kalau ada, load histori lama
#   3. Analisis transaksi baru + gabung histori
#   4. Update cache setelah selesai
# ─────────────────────────────────────────────────────────────

import json
import logging
import re
import threading
from datetime import datetime, date
from pathlib  import Path

logger     = logging.getLogger("fradara.cache")
CACHE_ROOT = Path("tmp")

# Lock per file untuk thread safety
_locks         : dict[str, threading.Lock] = {}
_lock_registry = threading.Lock()


def _get_lock(key: str) -> threading.Lock:
    with _lock_registry:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


def _slugify(name: str) -> str:
    """Ubah nama vendor/employee jadi filename yang aman."""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "_", slug)
    return slug[:80]


def _company_dir(company_id: str) -> Path:
    d = CACHE_ROOT / company_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _vendor_path(company_id: str, vendor_name: str) -> Path:
    return _company_dir(company_id) / f"vendor_{_slugify(vendor_name)}.json"


def _expense_path(company_id: str, employee_ref: str) -> Path:
    return _company_dir(company_id) / f"expense_{_slugify(employee_ref)}.json"


def _json_serial(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Cache read error %s: %s", path, e)
        return None


def _write(path: Path, data: dict) -> None:
    """Tulis JSON secara atomic (via temp file → rename)."""
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False,
                      indent=2, default=_json_serial)
        tmp_path.replace(path)
    except OSError as e:
        logger.error("Cache write error %s: %s", path, e)
        tmp_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────
# VENDOR CACHE
# ─────────────────────────────────────────────────────────────

def load_vendor_history(company_id: str, vendor_name: str) -> dict | None:
    """Load histori vendor dari cache. Return None jika belum ada."""
    path = _vendor_path(company_id, vendor_name)
    data = _read(path)
    if data:
        logger.info("[cache] Vendor HIT  company=%s vendor=%s tx=%d",
                    company_id, vendor_name, data.get("totalTransactions", 0))
    else:
        logger.info("[cache] Vendor MISS company=%s vendor=%s",
                    company_id, vendor_name)
    return data


def update_vendor_cache(
    company_id : str,
    vendor_name: str,
    new_tx     : dict,
    is_fraud   : bool,
    max_history: int = 50,
) -> dict:
    """Append transaksi baru ke cache vendor, buat baru jika belum ada."""
    path     = _vendor_path(company_id, vendor_name)
    lock_key = f"vendor:{company_id}:{_slugify(vendor_name)}"

    with _get_lock(lock_key):
        existing = _read(path) or {
            "vendorName"       : vendor_name,
            "companyId"        : company_id,
            "totalTransactions": 0,
            "totalValue"       : 0.0,
            "fraudCount"       : 0,
            "transactions"     : [],
            "createdAt"        : datetime.utcnow().isoformat(),
            "updatedAt"        : datetime.utcnow().isoformat(),
        }

        existing["totalTransactions"] += 1
        existing["totalValue"]        += float(new_tx.get("amountTotal", 0))
        if is_fraud:
            existing["fraudCount"] += 1
        existing["updatedAt"] = datetime.utcnow().isoformat()

        tx_record = {
            "purchaseId"  : new_tx.get("purchaseId") or new_tx.get("id"),
            "purchaseDate": new_tx.get("purchaseDate") or new_tx.get("transactionDate"),
            "amountTotal" : float(new_tx.get("amountTotal", 0)),
            "department"  : new_tx.get("department"),
            "status"      : new_tx.get("status", "Pending"),
            "isFraud"     : is_fraud,
            "processedAt" : datetime.utcnow().isoformat(),
        }
        existing["transactions"].append(tx_record)

        # Simpan hanya max_history transaksi terakhir
        if len(existing["transactions"]) > max_history:
            existing["transactions"] = existing["transactions"][-max_history:]

        _write(path, existing)
        logger.info("[cache] Vendor UPDATED company=%s vendor=%s tx=%d fraud=%d",
                    company_id, vendor_name,
                    existing["totalTransactions"], existing["fraudCount"])
        return existing


# ─────────────────────────────────────────────────────────────
# EXPENSE CACHE
# ─────────────────────────────────────────────────────────────

def load_expense_history(company_id: str, employee_ref: str) -> dict | None:
    """Load histori expense karyawan dari cache."""
    path = _expense_path(company_id, employee_ref)
    data = _read(path)
    if data:
        logger.info("[cache] Expense HIT  company=%s employee=%s tx=%d",
                    company_id, employee_ref, data.get("totalTransactions", 0))
    else:
        logger.info("[cache] Expense MISS company=%s employee=%s",
                    company_id, employee_ref)
    return data


def update_expense_cache(
    company_id  : str,
    employee_ref: str,
    department  : str | None,
    new_tx      : dict,
    is_fraud    : bool,
    max_history : int = 50,
) -> dict:
    """Append transaksi baru ke cache expense karyawan."""
    path     = _expense_path(company_id, employee_ref)
    lock_key = f"expense:{company_id}:{_slugify(employee_ref)}"

    with _get_lock(lock_key):
        existing = _read(path) or {
            "employeeExternalRef": employee_ref,
            "companyId"          : company_id,
            "department"         : department,
            "totalTransactions"  : 0,
            "totalValue"         : 0.0,
            "fraudCount"         : 0,
            "transactions"       : [],
            "createdAt"          : datetime.utcnow().isoformat(),
            "updatedAt"          : datetime.utcnow().isoformat(),
        }

        existing["totalTransactions"] += 1
        existing["totalValue"]        += float(new_tx.get("amountTotal", 0))
        if is_fraud:
            existing["fraudCount"] += 1
        if department:
            existing["department"] = department
        existing["updatedAt"] = datetime.utcnow().isoformat()

        tx_record = {
            "expenseId"  : new_tx.get("expenseId") or new_tx.get("id"),
            "expenseDate": new_tx.get("expenseDate") or new_tx.get("transactionDate"),
            "amountTotal": float(new_tx.get("amountTotal", 0)),
            "category"   : new_tx.get("category"),
            "merchant"   : new_tx.get("merchant"),
            "status"     : new_tx.get("status", "Pending"),
            "isFraud"    : is_fraud,
            "processedAt": datetime.utcnow().isoformat(),
        }
        existing["transactions"].append(tx_record)

        if len(existing["transactions"]) > max_history:
            existing["transactions"] = existing["transactions"][-max_history:]

        _write(path, existing)
        logger.info("[cache] Expense UPDATED company=%s employee=%s tx=%d fraud=%d",
                    company_id, employee_ref,
                    existing["totalTransactions"], existing["fraudCount"])
        return existing


# ─────────────────────────────────────────────────────────────
# KONVERSI CACHE → SCHEMA
# ─────────────────────────────────────────────────────────────

def _cache_to_vendor_history(cached: dict | None, vendor_name: str):
    """Ubah dict cache ke VendorHistory schema."""
    if not cached or not vendor_name:
        return None
    from schemas import VendorHistory, VendorHistoryItem
    items = []
    for t in cached.get("transactions", []):
        try:
            items.append(VendorHistoryItem(
                purchaseId   = str(t.get("purchaseId", "")),
                purchaseDate = str(t.get("purchaseDate", "2024-01-01"))[:10],
                amountTotal  = float(t.get("amountTotal", 0)),
                status       = str(t.get("status", "Approved")),
                department   = t.get("department"),
                isFraud      = bool(t.get("isFraud", False)),
            ))
        except Exception:
            continue
    return VendorHistory(
        vendorName        = vendor_name,
        totalTransactions = cached.get("totalTransactions", 0),
        totalValue        = cached.get("totalValue", 0.0),
        transactions      = items,
    )


def _cache_to_expense_history(cached: dict | None, employee_ref: str):
    """Ubah dict cache ke ExpenseHistory schema."""
    if not cached:
        return None
    from schemas import ExpenseHistory, ExpenseHistoryItem
    items = []
    for t in cached.get("transactions", []):
        try:
            items.append(ExpenseHistoryItem(
                expenseId   = str(t.get("expenseId", "")),
                expenseDate = str(t.get("expenseDate", "2024-01-01"))[:10],
                amountTotal = float(t.get("amountTotal", 0)),
                category    = t.get("category"),
                status      = str(t.get("status", "Approved")),
                isFraud     = bool(t.get("isFraud", False)),
            ))
        except Exception:
            continue
    return ExpenseHistory(
        employeeExternalRef = employee_ref,
        department          = cached.get("department"),
        totalTransactions   = cached.get("totalTransactions", 0),
        totalValue          = cached.get("totalValue", 0.0),
        transactions        = items,
    )


# ─────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────

def get_company_summary(company_id: str) -> dict:
    """Ringkasan semua cache untuk satu company."""
    d        = _company_dir(company_id)
    vendors  = [_read(f) for f in d.glob("vendor_*.json")]
    expenses = [_read(f) for f in d.glob("expense_*.json")]
    vendors  = [v for v in vendors if v]
    expenses = [e for e in expenses if e]

    return {
        "companyId"     : company_id,
        "totalVendors"  : len(vendors),
        "totalEmployees": len(expenses),
        "vendorSummary" : [
            {
                "vendorName"       : v["vendorName"],
                "totalTransactions": v["totalTransactions"],
                "totalValue"       : v["totalValue"],
                "fraudCount"       : v["fraudCount"],
                "updatedAt"        : v["updatedAt"],
            } for v in vendors
        ],
        "expenseSummary": [
            {
                "employeeRef"      : e["employeeExternalRef"],
                "department"       : e.get("department"),
                "totalTransactions": e["totalTransactions"],
                "totalValue"       : e["totalValue"],
                "fraudCount"       : e["fraudCount"],
                "updatedAt"        : e["updatedAt"],
            } for e in expenses
        ],
    }


def clear_company_cache(company_id: str) -> int:
    """Hapus semua cache untuk satu company. Return jumlah file dihapus."""
    d       = _company_dir(company_id)
    removed = 0
    for f in d.glob("*.json"):
        f.unlink(missing_ok=True)
        removed += 1
    logger.info("[cache] Cleared company=%s files=%d", company_id, removed)
    return removed