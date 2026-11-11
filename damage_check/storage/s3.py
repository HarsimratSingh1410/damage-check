"""S3 storage backend (Phase 2)."""

from __future__ import annotations

import io
import os
import tempfile
from typing import Optional

import cv2
import numpy as np

from ..core.interfaces import Storage


class S3Storage(Storage):
    """
    S3-backed storage using boto3.

    Configuration (via env vars or Settings):
      AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
      DAMAGE_S3_BUCKET   — target bucket name
      DAMAGE_S3_PREFIX   — optional key prefix, e.g. "damage-check/"
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        prefix: str = "",
        region: Optional[str] = None,
    ):
        try:
            import boto3
        except ImportError as e:
            raise ImportError(
                "boto3 is required for S3Storage. Install it with: pip install boto3"
            ) from e

        self.bucket = bucket or os.environ.get("DAMAGE_S3_BUCKET", "")
        if not self.bucket:
            raise ValueError(
                "S3 bucket name must be provided via constructor or DAMAGE_S3_BUCKET env var."
            )
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""
        self._s3 = boto3.client("s3", region_name=region)

    # ------------------------------------------------------------------
    # Storage interface
    # ------------------------------------------------------------------

    def save_image(self, image: np.ndarray, name: str) -> str:
        """Encode image as PNG and upload to S3. Returns the s3:// URL."""
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError(f"Failed to encode image '{name}' as PNG.")
        key = f"{self.prefix}{name}.png"
        self._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buf.tobytes(),
            ContentType="image/png",
        )
        return f"s3://{self.bucket}/{key}"

    def load_image(self, url: str) -> np.ndarray:
        """
        Download an image from S3 (s3://bucket/key) or an HTTPS URL
        and return it as a BGR uint8 ndarray.
        """
        if url.startswith("s3://"):
            bucket, key = self._parse_s3_url(url)
            resp = self._s3.get_object(Bucket=bucket, Key=key)
            data = resp["Body"].read()
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        elif url.startswith("http://") or url.startswith("https://"):
            import urllib.request

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                urllib.request.urlretrieve(url, f.name)
                img = cv2.imread(f.name)
            os.unlink(f.name)
        else:
            # Treat as a local path (useful during testing)
            img = cv2.imread(url)

        if img is None:
            raise FileNotFoundError(f"Could not load image from: {url}")
        return img

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_s3_url(url: str) -> tuple[str, str]:
        """Parse 's3://bucket/key/path' → ('bucket', 'key/path')."""
        without_scheme = url[len("s3://"):]
        bucket, _, key = without_scheme.partition("/")
        return bucket, key

    def generate_presigned_url(self, s3_url: str, expires_in: int = 3600) -> str:
        """Return a time-limited HTTPS URL for the given s3:// URL."""
        bucket, key = self._parse_s3_url(s3_url)
        return self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in,
        )
