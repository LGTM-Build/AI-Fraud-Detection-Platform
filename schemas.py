from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModuleType(str, Enum):
    PROCUREMENT = "procurement"
    EXPENSE = "expense"


class JobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"


class RiskLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    SAFE = "SAFE"


class AnalysisType(str, Enum):
    ANOMALY = "anomaly"


class IntentType(str, Enum):
    INFLATED_PRICE = "INFLATED_PRICE"
    DUPLICATE_VENDOR = "DUPLICATE_VENDOR"
    SPLIT_TRANSACTION = "SPLIT_TRANSACTION"
    WEEKEND_TRANSACTION = "WEEKEND_TRANSACTION"
    ABNORMAL_AMOUNT = "ABNORMAL_AMOUNT"
    SUSPICIOUS_VENDOR = "SUSPICIOUS_VENDOR"
    UNUSUAL_PATTERN = "UNUSUAL_PATTERN"
    NORMAL = "NORMAL"


class RequestMetadata(BaseModel):
    companyId: str | None = None
    requestedBy: str | None = None
    source: str | None = None


class CallbackHeaders(BaseModel):
    x_internal_api_key: str | None = Field(None, alias="x-internal-api-key")
    authorization: str | None = Field(None, alias="Authorization")

    model_config = {"populate_by_name": True}


class FraudAnalysisRequest(BaseModel):
    module: ModuleType
    callbackUrl: str
    callbackHeaders: CallbackHeaders | None = None
    metadata: RequestMetadata | None = None
    records: list[dict[str, Any]]


class AcceptedResponse(BaseModel):
    jobId: str
    message: str
    status: JobStatus
    module: ModuleType
    total: int


class JobInfo(BaseModel):
    jobId: str
    status: JobStatus
    module: ModuleType
    total: int
    createdAt: datetime
    updatedAt: datetime
    error: str | None = None


class ModelScores(BaseModel):
    isolationForest: float
    autoencoder: float
    ensemble: float
    fraudScore: float


class IntentResult(BaseModel):
    intent: IntentType
    confidence: float
    description: str


class TransactionResult(BaseModel):
    transactionId: str
    module: ModuleType
    fraudScore: float
    predictedFraud: bool
    riskLevel: RiskLevel
    scores: ModelScores
    intent: IntentResult
    reasons: list[str]
    amountTotal: float | None = None
    department: str | None = None
    vendorName: str | None = None
    merchant: str | None = None
    category: str | None = None
    transactionDate: str | None = None


class CallbackResultItem(BaseModel):
    module: ModuleType | None = None
    id: str | None = None
    procurementId: str | None = None
    purchaseId: str | None = None
    expenseDbId: str | None = None
    expenseId: str | None = None
    scores: dict[str, float] | None = None
    fraudScore: float | None = None
    riskLevel: RiskLevel | None = None
    predictedFraud: bool | None = None
    reasons: list[str] | None = None
    aiExplanation: str | None = None
    raw: dict[str, Any] | None = None


class BatchCallbackPayload(BaseModel):
    module: ModuleType
    generatedAt: datetime
    results: list[CallbackResultItem]
    samplePredictions: list[CallbackResultItem] | None = None


class VendorHistoryItem(BaseModel):
    purchaseId: str
    purchaseDate: str
    amountTotal: float
    status: str
    department: str | None = None
    isFraud: bool = False


class VendorHistory(BaseModel):
    vendorName: str
    totalTransactions: int
    totalValue: float
    transactions: list[VendorHistoryItem]


class ExpenseHistoryItem(BaseModel):
    expenseId: str
    expenseDate: str
    amountTotal: float
    category: str | None = None
    status: str
    isFraud: bool = False


class ExpenseHistory(BaseModel):
    employeeExternalRef: str
    department: str | None = None
    totalTransactions: int
    totalValue: float
    transactions: list[ExpenseHistoryItem]
