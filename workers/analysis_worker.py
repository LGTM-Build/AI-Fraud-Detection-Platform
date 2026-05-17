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
from datetime import datetime, timezone

from schemas import (
    BatchCallbackPayload,
    CallbackResultItem,
    FraudAnalysisRequest,
    JobStatus,
)
from services.normalizer          import normalize
from services.anomaly_detection_v2 import analyze
from clients.callback_client      import send_callback
from workers.job_store            import job_store

logger = logging.getLogger("fradara.worker")


def _build_callback_item(module: str, raw_record: dict, result) -> CallbackResultItem:
    scores = result.scores.model_dump()
    ai_explanation = ". ".join(result.reasons) if result.reasons else None

    base = {
        "module": module,
        "scores": scores,
        "fraudScore": result.fraudScore,
        "riskLevel": result.riskLevel,
        "predictedFraud": result.predictedFraud,
        "reasons": result.reasons,
        "aiExplanation": ai_explanation,
        "raw": {
            "analysis": result.model_dump(mode="json", exclude_none=True),
            "sourceRecord": raw_record,
        },
    }

    if module == "expense":
        return CallbackResultItem(
            **base,
            id=raw_record.get("id"),
            expenseDbId=raw_record.get("id"),
            expenseId=raw_record.get("expenseId"),
        )

    return CallbackResultItem(
        **base,
        id=raw_record.get("id"),
        procurementId=raw_record.get("id"),
        purchaseId=raw_record.get("purchaseId"),
    )


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
        callback_results = []

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
            callback_results.append(_build_callback_item(module, rec_dict, result))

        # ── Susun summary ─────────────────────────────────────
        elapsed       = round((time.time() - start) * 1000, 2)
        batch = BatchCallbackPayload(
            module=module,
            generatedAt=datetime.now(timezone.utc),
            results=callback_results,
            samplePredictions=callback_results[: min(3, len(callback_results))],
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
            payload          = batch.model_dump(mode="json", exclude_none=True),
        )

        final_status = JobStatus.DONE if ok else JobStatus.FAILED
        error_msg    = None if ok else "Callback ke Node.js gagal setelah semua retry"
        job_store.update(job_id, final_status, error=error_msg)

        logger.info(
            "[%s] Worker DONE module=%s total=%d fraud=%d "
            "time=%.1fms callback=%s",
            job_id, module, total, sum(1 for r in results if r.predictedFraud),
            elapsed, "OK" if ok else "FAIL",
        )

    except Exception as e:
        job_store.update(job_id, JobStatus.FAILED, error=str(e))
        logger.exception("[%s] Worker ERROR: %s", job_id, e)
