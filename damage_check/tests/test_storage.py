"""Tests for LocalStorage and S3Storage backend implementations."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from damage_check.storage.local import LocalStorage
from damage_check.storage.s3 import S3Storage


def _make_dummy_image(w=100, h=100):
    return np.full((h, w, 3), 120, dtype=np.uint8)


# ---------------------------------------------------------------------------
# LocalStorage tests
# ---------------------------------------------------------------------------

def test_local_storage_save_and_load(tmp_path):
    storage = LocalStorage(output_dir=str(tmp_path))
    img = _make_dummy_image()

    saved_path = storage.save_image(img, "test_img")
    assert os.path.exists(saved_path)

    loaded = storage.load_image(saved_path)
    assert loaded is not None
    assert loaded.shape == img.shape
    np.testing.assert_array_equal(loaded, img)


def test_local_storage_load_missing_file(tmp_path):
    storage = LocalStorage(output_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        storage.load_image(str(tmp_path / "nonexistent.png"))


# ---------------------------------------------------------------------------
# S3Storage tests
# ---------------------------------------------------------------------------

def test_s3_storage_requires_bucket(monkeypatch):
    monkeypatch.delenv("DAMAGE_S3_BUCKET", raising=False)
    with pytest.raises(ValueError, match="S3 bucket name must be provided"):
        S3Storage(bucket="")


def test_s3_storage_url_parsing():
    bucket, key = S3Storage._parse_s3_url("s3://my-bucket/path/to/image.png")
    assert bucket == "my-bucket"
    assert key == "path/to/image.png"


def test_s3_storage_save_image():
    mock_s3 = MagicMock()
    storage = S3Storage(bucket="test-bucket", prefix="runs/123", s3_client=mock_s3)
    img = _make_dummy_image()

    url = storage.save_image(img, "front_view")
    assert url == "s3://test-bucket/runs/123/front_view.png"

    mock_s3.put_object.assert_called_once()
    call_kwargs = mock_s3.put_object.call_args[1]
    assert call_kwargs["Bucket"] == "test-bucket"
    assert call_kwargs["Key"] == "runs/123/front_view.png"
    assert call_kwargs["ContentType"] == "image/png"


def test_s3_storage_load_image_from_s3():
    img = _make_dummy_image()
    _, buf = cv2.imencode(".png", img)

    mock_body = MagicMock()
    mock_body.read.return_value = buf.tobytes()

    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": mock_body}

    storage = S3Storage(bucket="test-bucket", s3_client=mock_s3)
    loaded = storage.load_image("s3://test-bucket/image.png")

    assert loaded is not None
    assert loaded.shape == img.shape
    mock_s3.get_object.assert_called_once_with(Bucket="test-bucket", Key="image.png")


def test_s3_storage_generate_presigned_url():
    mock_s3 = MagicMock()
    mock_s3.generate_presigned_url.return_value = "https://test-bucket.s3.amazonaws.com/image.png?sig=abc"

    storage = S3Storage(bucket="test-bucket", s3_client=mock_s3)
    url = storage.generate_presigned_url("s3://test-bucket/image.png", expires_in=1800)

    assert "https://" in url
    mock_s3.generate_presigned_url.assert_called_once_with(
        "get_object",
        Params={"Bucket": "test-bucket", "Key": "image.png"},
        ExpiresIn=1800,
    )

