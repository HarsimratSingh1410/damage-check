"""FastAPI async job service for the Damage-Check Module."""

from __future__ import annotations

import asyncio
import uuid
from typing import Dict, List, Optional

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..core.models import JobInput, JobResult, JobStatus
from ..config import settings as cfg

app = FastAPI(
    title="Damage-Check Service",
    version="1.0.0",
    description=(
        "Async job API for before/after damage detection. "
        "Submit a job, poll for results, or receive a webhook callback."
    ),
)

# In-memory job store (swap for Redis / PostgreSQL in production)
_jobs: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class WebhookConfig(BaseModel):
    callback_url: Optional[str] = None


class SubmitRequest(BaseModel):
    job: JobInput
    webhook: Optional[WebhookConfig] = None


class JobSummary(BaseModel):
    job_id: str
    status: str
    item_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Liveness check."""
    return {"status": "ok", "version": app.version}


@app.post("/jobs", status_code=202, response_model=JobSummary)
async def submit_job(request: SubmitRequest, background_tasks: BackgroundTasks):
    """
    Submit a damage-check job.

    Returns `job_id` and `status: pending` immediately.
    Use `GET /jobs/{job_id}` to poll for the result.
    """
    job_id = request.job.job_id or str(uuid.uuid4())
    request.job.job_id = job_id
    _jobs[job_id] = {
        "status": "pending",
        "result": None,
        "item_id": request.job.item_id,
    }
    background_tasks.add_task(_run_job, request.job, request.webhook)
    return JobSummary(job_id=job_id, status="pending", item_id=request.job.item_id)


@app.get("/jobs", response_model=List[JobSummary])
async def list_jobs():
    """List all submitted jobs with their current status."""
    return [
        JobSummary(job_id=jid, status=entry["status"], item_id=entry.get("item_id"))
        for jid, entry in _jobs.items()
    ]


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """
    Get job status and full results.

    While pending/running returns `{job_id, status}`.
    When complete returns the full `JobResult` schema.
    When errored returns `{job_id, status: "error", error: "..."}`.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    entry = _jobs[job_id]
    if entry["result"]:
        return entry["result"]
    return {"job_id": job_id, "status": entry["status"], "error": entry.get("error")}


@app.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str):
    """Remove a completed job from the in-memory store."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if _jobs[job_id]["status"] in ("pending", "running"):
        raise HTTPException(status_code=409, detail="Cannot delete a job that is still running.")
    del _jobs[job_id]


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

async def _run_job(job: JobInput, webhook: Optional[WebhookConfig]) -> None:
    """Run the pipeline in a thread pool executor (keeps the event loop free)."""
    _jobs[job.job_id]["status"] = "running"
    try:
        loop = asyncio.get_running_loop()
        result: JobResult = await loop.run_in_executor(None, _sync_run, job)
        _jobs[job.job_id]["status"] = result.status.value
        _jobs[job.job_id]["result"] = result.model_dump(mode="json")
        if webhook and webhook.callback_url:
            await _notify_webhook(webhook.callback_url, result)
    except Exception as exc:
        _jobs[job.job_id]["status"] = "error"
        _jobs[job.job_id]["error"] = str(exc)


def get_pipeline():
    """Build the pipeline from config."""
    from ..core.pipeline import Pipeline

    # Storage backend
    if cfg.storage_backend == "s3":
        from ..storage.s3 import S3Storage
        storage = S3Storage()
    else:
        from ..storage.local import LocalStorage
        storage = LocalStorage()

    # Segmenter
    if cfg.segmenter == "rembg":
        from ..stages.segment_rembg import RembgSegmenter
        segmenter = RembgSegmenter()
    elif cfg.segmenter == "sam":
        from ..stages.segment_sam import SAMSegmenter
        segmenter = SAMSegmenter()
    elif cfg.segmenter in ("none", "passthrough", "dummy"):
        from ..core.interfaces import Segmenter
        class PassthroughSegmenter(Segmenter):
            def segment(self, image: np.ndarray) -> np.ndarray:
                return np.ones(image.shape[:2], dtype=np.uint8) * 255
        segmenter = PassthroughSegmenter()
    else:
        raise ValueError(f"Unknown segmenter: {cfg.segmenter!r}")

    # Aligner
    if cfg.aligner == "sift":
        from ..stages.align_sift import SIFTAligner
        aligner = SIFTAligner()
    elif cfg.aligner == "loftr":
        from ..stages.align_loftr import LoFTRAligner
        aligner = LoFTRAligner()
    else:
        raise ValueError(f"Unknown aligner: {cfg.aligner!r}")

    # Comparator
    if cfg.comparator == "ssim":
        from ..stages.compare_ssim import SSIMComparator
        comparator = SSIMComparator()
    elif cfg.comparator == "dino":
        from ..stages.compare_dino import DINOv2Comparator
        comparator = DINOv2Comparator()
    else:
        raise ValueError(f"Unknown comparator: {cfg.comparator!r}")

    # Optional reviewer
    reviewer = None
    if cfg.enable_vlm:
        from ..stages.review_vlm import VLMReviewer
        reviewer = VLMReviewer()

    return Pipeline(
        segmenter=segmenter,
        aligner=aligner,
        comparator=comparator,
        storage=storage,
        reviewer=reviewer,
    )


def _sync_run(job: JobInput) -> JobResult:
    """Run pipeline synchronously."""
    pipeline = get_pipeline()
    return pipeline.run(job)


async def _notify_webhook(url: str, result: JobResult) -> None:
    """Best-effort webhook delivery — never raises."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            await client.post(url, json=result.model_dump(mode="json"), timeout=10)
    except Exception:
        pass
