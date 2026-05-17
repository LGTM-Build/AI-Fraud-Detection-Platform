# schemas.py
# ─────────────────────────────────────────────────────────────
# Semua Pydantic schema untuk Fradara Fraud Detection Service
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
from datetime   import datetime
from enum       import Enum
from typing     import Any, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────

class ModuleType(str, Enum):
    PROCUREMENT = "procurement"
    EXPENSE     = "expense"


class JobStatus(str, Enum):
    PENDING    = "PENDING"
    PROCESSING = "PROCESSING"
    DONE       = "DONE"
    FAILED     = "FAILED"


class RiskLevel(str, Enum):
    HIGH   = "HIGH"    # 75–100
    MEDIUM = "MEDIUM"  # 45–74
    LOW    = "LOW"     # 20–44
    SAFE   = "SAFE"    # 0–19


class AnalysisType(str, Enum):
    ANOMALY = "anomaly"


class IntentType(str, Enum):
    INFLATED_PRICE      = "INFLATED_PRICE"
    DUPLICATE_VENDOR    = "DUPLICATE_VENDOR"
    SPLIT_TRANSACTION   = "SPLIT_TRANSACTION"
    WEEKEND_TRANSACTION = "WEEKEND_TRANSACTION"
    ABNORMAL_AMOUNT     = "ABNORMAL_AMOUNT"
    SUSPICIOUS_VENDOR   = "SUSPICIOUS_VENDOR"
    UNUSUAL_PATTERN     = "UNUSUAL_PATTERN"
    NORMAL              = "NORMAL"


# ─────────────────────────────────────────────────────────────
# REQUEST — Record Procurement
# ─────────────────────────────────────────────────────────────

class ProcurementRecord(BaseModel):
    purchaseId             : str
    transactionType        : Optional[str]   = "Procurement"
    employeeId             : Optional[str]   = None
    department             : Optional[str]   = None
    location               : Optional[str]   = None
    purchaseDate           : Optional[str]   = None
    approvalDate           : Optional[str]   = None
    invoiceNumber          : Optional[str]   = None
    invoiceDate            : Optional[str]   = None
    paymentDate            : Optional[str]   = None
    vendorName             : Optional[str]   = None
    vendorRegistrationDate : Optional[str]   = None
    contractDate           : Optional[str]   = None
    vendorBankAccount      : Optional[str]   = None
    vendorAddress          : Optional[str]   = None
    vendorContact          : Optional[str]   = None
    itemId                 : Optional[str]   = None
    itemDescription        : Optional[str]   = None
    category               : Optional[str]   = None
    unitPrice              : Optional[float] = 0.0
    quantity               : Optional[float] = 1.0
    amountTotal            : Optional[float] = 0.0
    status                 : Optional[str]   = "Pending"
    contractId             : Optional[str]   = None


# ─────────────────────────────────────────────────────────────
# REQUEST — Record Expense
# ─────────────────────────────────────────────────────────────

class ExpenseRecord(BaseModel):
    expenseId      : str
    employeeId     : Optional[str]   = None
    department     : Optional[str]   = None
    location       : Optional[str]   = None
    expenseDate    : Optional[str]   = None
    approvalDate   : Optional[str]   = None
    merchant       : Optional[str]   = None
    category       : Optional[str]   = None
    itemDescription: Optional[str]   = None
    unitPrice      : Optional[float] = 0.0
    quantity       : Optional[float] = 1.0
    amountTotal    : Optional[float] = 0.0
    status         : Optional[str]   = "Pending"
    paymentMethod  : Optional[str]   = None
    receiptNumber  : Optional[str]   = None


# ─────────────────────────────────────────────────────────────
# REQUEST — Metadata & Headers
# ─────────────────────────────────────────────────────────────

class RequestMetadata(BaseModel):
    companyId  : Optional[str] = None
    requestedBy: Optional[str] = None
    source     : Optional[str] = None


class CallbackHeaders(BaseModel):
    x_api_key    : Optional[str] = Field(None, alias="x-api-key")
    authorization: Optional[str] = Field(None, alias="Authorization")

    model_config = {"populate_by_name": True}


# ─────────────────────────────────────────────────────────────
# REQUEST — Main Request Body
# ─────────────────────────────────────────────────────────────

class FraudAnalysisRequest(BaseModel):
    module         : ModuleType
    callbackUrl    : str
    callbackHeaders: Optional[CallbackHeaders] = None
    metadata       : Optional[RequestMetadata] = None
    records        : list[dict[str, Any]]


# ─────────────────────────────────────────────────────────────
# RESPONSE — Job
# ─────────────────────────────────────────────────────────────

class AcceptedResponse(BaseModel):
    jobId  : str
    message: str
    status : JobStatus
    module : ModuleType
    total  : int


class JobInfo(BaseModel):
    jobId    : str
    status   : JobStatus
    module   : ModuleType
    total    : int
    createdAt: datetime
    updatedAt: datetime
    error    : Optional[str] = None


# ─────────────────────────────────────────────────────────────
# ANALYSIS RESULT — Per Transaksi
# ─────────────────────────────────────────────────────────────

class ModelScores(BaseModel):
    isolationForest: float
    autoencoder    : float
    ensemble       : float
    fraudScore     : float


class IntentResult(BaseModel):
    intent     : IntentType
    confidence : float
    description: str


class TransactionResult(BaseModel):
    # Identitas transaksi
    transactionId  : str
    module         : ModuleType

    # Skor & prediksi
    fraudScore     : float          # 0–100
    predictedFraud : bool
    riskLevel      : RiskLevel
    scores         : ModelScores

    # Intent & alasan
    intent         : IntentResult
    reasons        : list[str]

    # Info tambahan
    amountTotal    : Optional[float] = None
    department     : Optional[str]   = None
    vendorName     : Optional[str]   = None   # procurement
    merchant       : Optional[str]   = None   # expense
    category       : Optional[str]   = None
    transactionDate: Optional[str]   = None


# ─────────────────────────────────────────────────────────────
# CALLBACK PAYLOAD — Dikirim ke Node.js
# ─────────────────────────────────────────────────────────────

class BatchCallbackPayload(BaseModel):
    jobId            : str
    module           : ModuleType
    analysisType     : AnalysisType
    total            : int
    fraudDetected    : int
    fraudRate        : float
    processingTimeMs : float
    summary          : dict[str, Any]
    results          : list[TransactionResult]
    requestMetadata  : Optional[RequestMetadata] = None
    analyzedAt       : datetime


# ─────────────────────────────────────────────────────────────
# CACHE SCHEMA — Dipakai di company_cache.py
# ─────────────────────────────────────────────────────────────

class VendorHistoryItem(BaseModel):
    purchaseId  : str
    purchaseDate: str
    amountTotal : float
    status      : str
    department  : Optional[str] = None
    isFraud     : bool = False


class VendorHistory(BaseModel):
    vendorName        : str
    totalTransactions : int
    totalValue        : float
    transactions      : list[VendorHistoryItem]


class ExpenseHistoryItem(BaseModel):
    expenseId  : str
    expenseDate: str
    amountTotal: float
    category   : Optional[str] = None
    status     : str
    isFraud    : bool = False


class ExpenseHistory(BaseModel):
    employeeExternalRef: str
    department         : Optional[str] = None
    totalTransactions  : int
    totalValue         : float
    transactions       : list[ExpenseHistoryItem]