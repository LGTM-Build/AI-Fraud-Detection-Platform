# workers/job_store.py
# ─────────────────────────────────────────────────────────────
# In-memory job store untuk tracking status background job.
# Thread-safe menggunakan Lock.
#
# Mudah diganti Redis/DB di production nanti —
# cukup ganti implementasi _store tanpa ubah interface.
# ─────────────────────────────────────────────────────────────

import logging
from collections import OrderedDict
from datetime    import datetime
from threading   import Lock

from config  import get_settings
from schemas import JobInfo, JobStatus, ModuleType

logger   = logging.getLogger("fradara.jobstore")
settings = get_settings()


class InMemoryJobStore:
    """
    Simple in-memory store untuk job status.
    Otomatis buang job paling lama jika melebihi MAX_JOBS_IN_MEMORY (FIFO).
    """

    def __init__(self, max_size: int = 1000):
        self._store   : OrderedDict[str, JobInfo] = OrderedDict()
        self._lock    : Lock = Lock()
        self._max_size: int  = max_size

    def create(
        self,
        job_id : str,
        module : ModuleType,
        total  : int = 0,
    ) -> JobInfo:
        """Buat job baru dengan status PENDING."""
        job = JobInfo(
            jobId     = job_id,
            status    = JobStatus.PENDING,
            module    = module,
            total     = total,
            createdAt = datetime.utcnow(),
            updatedAt = datetime.utcnow(),
        )
        with self._lock:
            # Buang job paling lama jika sudah penuh
            if len(self._store) >= self._max_size:
                oldest = next(iter(self._store))
                self._store.pop(oldest)
                logger.debug("Job store penuh — buang job lama: %s", oldest)
            self._store[job_id] = job

        logger.info("Job dibuat jobId=%s module=%s total=%d",
                    job_id, module, total)
        return job

    def update(
        self,
        job_id: str,
        status: JobStatus,
        error : str | None = None,
    ) -> None:
        """Update status job."""
        with self._lock:
            if job_id in self._store:
                self._store[job_id].status    = status
                self._store[job_id].updatedAt = datetime.utcnow()
                if error:
                    self._store[job_id].error = error

        logger.info("Job updated jobId=%s status=%s", job_id, status)

    def get(self, job_id: str) -> JobInfo | None:
        """Ambil job berdasarkan jobId."""
        with self._lock:
            return self._store.get(job_id)

    def all(self) -> list[JobInfo]:
        """Ambil semua job."""
        with self._lock:
            return list(self._store.values())


# Singleton — dipakai di seluruh aplikasi
job_store = InMemoryJobStore(max_size=settings.MAX_JOBS_IN_MEMORY)