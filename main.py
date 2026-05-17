# main.py
# ─────────────────────────────────────────────────────────────
# Fradara Fraud Analysis Service — Entry Point
#
# Cara menjalankan:
#   uvicorn main:app --reload --host 0.0.0.0 --port 8000
#
# Swagger UI (dokumentasi interaktif):
#   http://localhost:8000/docs
# ─────────────────────────────────────────────────────────────

import logging
import uuid
from datetime import datetime

from fastapi                 import (
    FastAPI, BackgroundTasks,
    HTTPException, Security, Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security        import APIKeyHeader

from config  import get_settings
from schemas import (
    FraudAnalysisRequest,
    AcceptedResponse,
    JobStatus,
    JobInfo,
)
from workers.job_store       import job_store
from workers.analysis_worker import process_request
from cache.company_cache     import (
    get_company_summary,
    clear_company_cache,
)

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
)
logger   = logging.getLogger("fradara.main")
settings = get_settings()

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title       = settings.APP_NAME,
    version     = settings.APP_VERSION,
    description = (
        "Fradara — AI Fraud Detection Service. "
        "Model: Isolation Forest + Autoencoder Ensemble (Anomaly Detection). "
        "Mendukung module: procurement, expense."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── API Key Auth ──────────────────────────────────────────────
api_key_header = APIKeyHeader(name="x-api-key", auto_error=True)

async def verify_api_key(key: str = Security(api_key_header)):
    """
    Validasi API key dari Node.js.
    Node.js wajib kirim header: x-api-key: <PYTHON_API_KEY>
    """
    if key != settings.PYTHON_API_KEY:
        logger.warning("Request ditolak — API key tidak valid")
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key

AUTH = [Depends(verify_api_key)]

# ─────────────────────────────────────────────────────────────
# ENDPOINTS — Info
# ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    """Health check & info service."""
    return {
        "status"          : "ok",
        "service"         : settings.APP_NAME,
        "version"         : settings.APP_VERSION,
        "model"           : "Isolation Forest (30%) + Autoencoder (70%) Ensemble",
        "mode"            : "real_model" if settings.USE_REAL_MODEL else "mock",
        "timestamp"       : datetime.utcnow().isoformat(),
        "supportedModules": ["procurement", "expense"],
        "endpoints"       : {
            "POST /analyze"                   : "Analisis transaksi (procurement / expense)",
            "GET  /jobs/{jobId}"              : "Cek status job",
            "GET  /cache/{companyId}/summary" : "Ringkasan cache perusahaan",
            "DELETE /cache/{companyId}"       : "Reset cache perusahaan",
            "GET  /docs"                      : "Swagger UI — dokumentasi interaktif",
        },
    }


@app.get("/health", tags=["Info"])
def health():
    """Health check detail."""
    return {
        "status"      : "healthy",
        "useRealModel": settings.USE_REAL_MODEL,
        "modelsDir"   : settings.MODELS_DIR,
        "totalJobs"   : len(job_store.all()),
        "nodeJsUrl"   : settings.NODEJS_BASE_URL,
    }


# ─────────────────────────────────────────────────────────────
# ENDPOINTS — Jobs
# ─────────────────────────────────────────────────────────────

@app.get(
    "/jobs/{job_id}",
    response_model = JobInfo,
    tags           = ["Jobs"],
    dependencies   = AUTH,
)
def get_job(job_id: str):
    """
    Cek status job berdasarkan jobId.
    Node.js bisa polling endpoint ini jika perlu.
    """
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(
            status_code = 404,
            detail      = f"Job {job_id} tidak ditemukan",
        )
    return job


# ─────────────────────────────────────────────────────────────
# ENDPOINTS — Analyze
# ─────────────────────────────────────────────────────────────

@app.post(
    "/analyze",
    response_model = AcceptedResponse,
    status_code    = 202,
    tags           = ["Analyze"],
    dependencies   = AUTH,
)
async def analyze(
    request         : FraudAnalysisRequest,
    background_tasks: BackgroundTasks,
):
    """
    Entry point utama dari Node.js.

    **Alur:**
    1. Validasi payload (Pydantic otomatis)
    2. Buat jobId unik
    3. Simpan job dengan status PENDING
    4. Balas **202 Accepted** langsung (tidak tunggu analisis selesai)
    5. Proses analisis di background
    6. Setelah selesai, Python callback ke `callbackUrl` dari payload

    **Module yang didukung:**
    - `procurement` → record berisi purchaseId, vendorName, dll
    - `expense`     → record berisi expenseId, category, merchant, dll

    **companyId di metadata** dipakai sebagai scope cache histori.
    Makin banyak transaksi diproses, makin akurat analisisnya.
    """
    job_id     = str(uuid.uuid4())
    total      = len(request.records)
    company_id = request.metadata.companyId if request.metadata else None

    job_store.create(job_id, module=request.module, total=total)

    logger.info(
        "Job diterima jobId=%s module=%s records=%d "
        "company=%s callbackUrl=%s",
        job_id, request.module, total,
        company_id, request.callbackUrl,
    )

    # Jalankan di background — request langsung dibalas
    background_tasks.add_task(process_request, job_id, request)

    return AcceptedResponse(
        jobId   = job_id,
        message = f"{request.module} analysis accepted ({total} records)",
        status  = JobStatus.PENDING,
        module  = request.module,
        total   = total,
    )


# ─────────────────────────────────────────────────────────────
# ENDPOINTS — Cache
# ─────────────────────────────────────────────────────────────

@app.get(
    "/cache/{company_id}/summary",
    tags        = ["Cache"],
    dependencies= AUTH,
)
def cache_summary(company_id: str):
    """
    Ringkasan cache untuk satu perusahaan.
    Menampilkan berapa vendor dan karyawan yang sudah ter-cache.
    """
    return get_company_summary(company_id)


@app.delete(
    "/cache/{company_id}",
    tags        = ["Cache"],
    dependencies= AUTH,
)
def cache_clear(company_id: str):
    """
    Hapus semua cache untuk satu perusahaan.
    Gunakan jika perlu reset histori dari awal.
    """
    removed = clear_company_cache(company_id)
    logger.info("Cache cleared company=%s files=%d", company_id, removed)
    return {
        "status"   : "ok",
        "companyId": company_id,
        "removed"  : removed,
        "message"  : f"{removed} cache file dihapus",
    }