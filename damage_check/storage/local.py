"""Local filesystem storage backend."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from ..core.interfaces import Storage
from ..config import settings as cfg


class LocalStorage(Storage):
    """Save and load images from the local filesystem."""

    def __init__(self, output_dir: str | None = None) -> None:
        self.output_dir = Path(output_dir or cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_image(self, image: np.ndarray, name: str) -> str:
        path = self.output_dir / f"{name}.png"
        cv2.imwrite(str(path), image)
        return str(path)

    def load_image(self, url: str) -> np.ndarray:
        if url.startswith("http://") or url.startswith("https://"):
            import urllib.request

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                urllib.request.urlretrieve(url, f.name)
                tmp_path = f.name
            img = cv2.imread(tmp_path)
            os.unlink(tmp_path)
        else:
            img = cv2.imread(url)

        if img is None:
            raise FileNotFoundError(f"Could not load image: {url}")
        return img
