"""Tests for the FastAPI Damage-Check service (Phase 2)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from damage_check.core.models import (
    JobInput,
    JobResult,
    JobStatus,
    QualityInfo,
    ViewInput,
    ViewResult,
)
from damage_check.service.api import app, _jobs


@pytest.fixture(autouse=True)
def clear_jobs():
    """Clear in-memory jobs store before each test."""
    _jobs.clear()
    yield
    _jobs.clear()


@pytest.fixture()
def client():
    return TestClient(app)


def _make_dummy_job_input(job_id: str = "test-job-1") -> dict:
    return {
        "job": {
            "schema_version": "1",
            "job_id": job_id,
            "item_id": "laptop-42",
            "views": [
                {
                    "view_name": "front",
                    "before_url": "before.png",
                    "after_url": "after.png",
                }
            ],
        }
    }


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_submit_job_and_poll(client):
    mock_result = JobResult(
        job_id="test-job-1",
        status=JobStatus.ok,
        views=[
            ViewResult(
                view_name="front",
                quality=QualityInfo(aligned=True, alignment_score=0.95, issues=[]),
                regions=[],
                heatmap_url="output/test_heatmap.png",
            )
        ],
    )

    with patch("damage_check.service.api._sync_run", return_value=mock_result):
        # 1. Submit job
        payload = _make_dummy_job_input("test-job-1")
        submit_resp = client.post("/jobs", json=payload)
        assert submit_resp.status_code == 202
        submit_data = submit_resp.json()
        assert submit_data["job_id"] == "test-job-1"
        assert submit_data["status"] == "pending"

        # In TestClient, BackgroundTasks run during the request cycle
        # 2. Get job results
        get_resp = client.get("/jobs/test-job-1")
        assert get_resp.status_code == 200
        get_data = get_resp.json()
        assert get_data["job_id"] == "test-job-1"
        assert get_data["status"] == "ok"
        assert len(get_data["views"]) == 1
        assert get_data["views"][0]["view_name"] == "front"


def test_list_jobs(client):
    mock_result = JobResult(
        job_id="job-abc",
        status=JobStatus.ok,
        views=[],
    )

    with patch("damage_check.service.api._sync_run", return_value=mock_result):
        client.post("/jobs", json=_make_dummy_job_input("job-abc"))

        resp = client.get("/jobs")
        assert resp.status_code == 200
        jobs_list = resp.json()
        assert len(jobs_list) == 1
        assert jobs_list[0]["job_id"] == "job-abc"
        assert jobs_list[0]["status"] == "ok"


def test_job_not_found(client):
    resp = client.get("/jobs/nonexistent-id")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_delete_completed_job(client):
    mock_result = JobResult(
        job_id="to-delete",
        status=JobStatus.ok,
        views=[],
    )

    with patch("damage_check.service.api._sync_run", return_value=mock_result):
        client.post("/jobs", json=_make_dummy_job_input("to-delete"))

        # Verify job is there
        assert client.get("/jobs/to-delete").status_code == 200

        # Delete job
        del_resp = client.delete("/jobs/to-delete")
        assert del_resp.status_code == 204

        # Now should be 404
        assert client.get("/jobs/to-delete").status_code == 404


def test_delete_nonexistent_job(client):
    resp = client.delete("/jobs/unknown")
    assert resp.status_code == 404


def test_webhook_delivery(client):
    mock_result = JobResult(
        job_id="webhook-job",
        status=JobStatus.flagged,
        views=[],
    )

    with patch("damage_check.service.api._sync_run", return_value=mock_result), \
         patch("damage_check.service.api._notify_webhook", new_callable=AsyncMock) as mock_webhook:
        
        payload = _make_dummy_job_input("webhook-job")
        payload["webhook"] = {"callback_url": "https://myapp.example.com/callback"}

        resp = client.post("/jobs", json=payload)
        assert resp.status_code == 202

        # Verify webhook was called with callback_url and result
        mock_webhook.assert_called_once()
        args = mock_webhook.call_args[0]
        assert args[0] == "https://myapp.example.com/callback"
        assert args[1].job_id == "webhook-job"

