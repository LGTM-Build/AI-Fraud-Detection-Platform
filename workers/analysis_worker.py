# workers/analysis_worker.py
# ─────────────────────────────────────────────────────────────
# Background worker — proses analisis fraud di background
# setelah request diterima, lalu callback ke Node.js.
#
# Alur:
#   1. Update status → PROCESSING
#   2. Normalisasi record (procurement / expense)
#   3. Analisis fraud via anomaly_detection_v2
#   4. Susun BatchCallbackPayload
#   5. Callback ke callbackUrl dari payload
#   6. Update status → DONE / FAILED
# ─────────────────────────────────────────────────────────────

import asyncio
import logging
import time
from datetime import datetime

from schemas import (
    FraudAnalysisRequest, BatchCallbackPayload,
    AnalysisType, JobStatus, ModuleType,
)
from services.normalizer          import normalize
from services.anomaly_detection_v2 import analyze
from clients.callback_client      import send_callback
from workers.job_store            import job_store

logger = logging.getLogger("fradara.worker")


async def process_request(job_id: str, request: FraudAnalysisRequest) -> None:
    """
    Background worker utama.
    Dipanggil via BackgroundTasks di main.py setelah request diterima.
    """
    total      = len(request.records)
    module     = request.module
    company_id = request.metadata.companyId if request.metadata else None

    logger.info(
        "[%s] Worker START module=%s records=%d company=%s",
        job_id, module, total, company_id,
    )
    job_store.update(job_id, JobStatus.PROCESSING)

    start = time.time()
    try:
        loop    = asyncio.get_event_loop()
        results = []

        for raw in request.records:
            # raw bisa dict atau Pydantic model
            rec_dict = raw if isinstance(raw, dict) else raw.model_dump()

            # Normalisasi → format internal
            tx = normalize(module, rec_dict)

            # Jalankan analisis di thread pool (CPU-bound)
            result = await loop.run_in_executor(
                None, analyze, job_id, tx, company_id
            )
            results.append(result)

        # ── Susun summary ─────────────────────────────────────
        elapsed       = round((time.time() - start) * 1000, 2)
        fraud_detected= sum(1 for r in results if r.predictedFraud)
        risk_counts   = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "SAFE": 0}
        intent_dist   = {}

        for r in results:
            risk_counts[r.riskLevel.value] += 1
            k = r.intent.intent.value
            intent_dist[k] = intent_dist.get(k, 0) + 1

        avg_score = round(
            sum(r.fraudScore for r in results) / max(total, 1), 2
        )

        batch = BatchCallbackPayload(
            jobId            = job_id,
            module           = module,
            analysisType     = AnalysisType.ANOMALY,
            total            = total,
            fraudDetected    = fraud_detected,
            fraudRate        = round(fraud_detected / total * 100, 2),
            processingTimeMs = elapsed,
            summary          = {
                "riskBreakdown": risk_counts,
                "intentDist"   : intent_dist,
                "avgFraudScore": avg_score,
                "companyId"    : company_id,
                "cacheUsed"    : company_id is not None,
            },
            results         = results,
            requestMetadata = request.metadata,
            analyzedAt      = datetime.utcnow(),
        )

        # ── Callback ke Node.js ───────────────────────────────
        cb_headers = {}
        if request.callbackHeaders:
            cb_headers = request.callbackHeaders.model_dump(
                by_alias=True, exclude_none=True
            )

        ok = await send_callback(
            callback_url     = request.callbackUrl,
            callback_headers = cb_headers,
            payload          = batch.model_dump(mode="json"),
        )

        final_status = JobStatus.DONE if ok else JobStatus.FAILED
        error_msg    = None if ok else "Callback ke Node.js gagal setelah semua retry"
        job_store.update(job_id, final_status, error=error_msg)

        logger.info(
            "[%s] Worker DONE module=%s total=%d fraud=%d "
            "time=%.1fms callback=%s",
            job_id, module, total, fraud_detected,
            elapsed, "OK" if ok else "FAIL",
        )

    except Exception as e:
        job_store.update(job_id, JobStatus.FAILED, error=str(e))
        logger.exception("[%s] Worker ERROR: %s", job_id, e)