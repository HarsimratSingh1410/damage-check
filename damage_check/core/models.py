"""Pydantic v2 schemas for the Damage-Check Module (Schema v1)."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class ViewInput(BaseModel):
    view_name: str
    before_url: str  # local path or http URL
    after_url: str


class JobInput(BaseModel):
    schema_version: str = "1"
    job_id: str
    item_id: str
    views: List[ViewInput]


class QualityInfo(BaseModel):
    aligned: bool
    alignment_score: float
    issues: List[str]


class DamageRegion(BaseModel):
    bbox: List[int]  # [x, y, w, h]
    score: float
    vlm_label: Optional[str] = None
    crop_before_url: Optional[str] = None
    crop_after_url: Optional[str] = None


class ViewResult(BaseModel):
    view_name: str
    quality: QualityInfo
    regions: List[DamageRegion]
    heatmap_url: Optional[str] = None


class JobStatus(str, Enum):
    ok = "ok"
    flagged = "flagged"
    needs_retake = "needs_retake"
    low_confidence = "low_confidence"


class JobResult(BaseModel):
    schema_version: str = "1"
    job_id: str
    status: JobStatus
    views: List[ViewResult]
