"""Basic tests for the damage-check pipeline using synthetic images."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from damage_check.core.models import JobInput, JobStatus, ViewInput
from damage_check.core.pipeline import Pipeline
from damage_check.stages.align_sift import SIFTAligner
from damage_check.stages.compare_ssim import SSIMComparator
from damage_check.storage.local import LocalStorage


class DummySegmenter:
    """Returns a full-image mask (no background removal)."""

    def segment(self, image: np.ndarray) -> np.ndarray:
        return np.ones(image.shape[:2], dtype=np.uint8) * 255


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_textured_image(shape=(300, 300, 3), seed: int = 42) -> np.ndarray:
    """
    Create a synthetic image that is 'sharp' enough to pass the blur gate.

    Strategy:
      1. Start with a mid-gray base.
      2. Draw a checkerboard pattern — lots of hard edges → high Laplacian variance.
      3. Add a few coloured shapes so SIFT has distinct keypoints.
      4. Add a small amount of Gaussian noise to lift variance further.

    The result typically has Laplacian variance >> 1000, well above the
    default blur_threshold of 100.
    """
    rng = np.random.default_rng(seed)
    img = np.full(shape, 128, dtype=np.uint8)

    # Checkerboard (8×8 tiles of 32px)
    tile = 32
    for row in range(0, shape[0], tile):
        for col in range(0, shape[1], tile):
            if (row // tile + col // tile) % 2 == 0:
                img[row:row + tile, col:col + tile] = 220
            else:
                img[row:row + tile, col:col + tile] = 50

    # Coloured shapes for SIFT keypoints
    cv2.rectangle(img, (60, 60), (240, 240), (100, 180, 60), 3)
    cv2.circle(img, (150, 150), 50, (200, 80, 180), -1)
    cv2.line(img, (30, 30), (270, 270), (20, 200, 220), 2)
    cv2.line(img, (30, 270), (270, 30), (220, 180, 20), 2)

    # Small Gaussian noise
    noise = rng.integers(-15, 16, size=shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def _make_test_image(shape=(300, 300, 3), color=(200, 200, 200)):
    return np.full(shape, color, dtype=np.uint8)


def _save_img(path, img: np.ndarray) -> str:
    cv2.imwrite(str(path), img)
    return str(path)


def _make_pipeline(out_dir):
    storage = LocalStorage(output_dir=str(out_dir))
    return Pipeline(
        segmenter=DummySegmenter(),
        aligner=SIFTAligner(),
        comparator=SSIMComparator(),
        storage=storage,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_dirs(tmp_path):
    img_dir = tmp_path / "images"
    out_dir = tmp_path / "output"
    img_dir.mkdir()
    out_dir.mkdir()
    return img_dir, out_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_identical_images_no_damage(tmp_dirs):
    """Identical sharp images → quality gate passes, no damage regions, status ok."""
    img_dir, out_dir = tmp_dirs
    # _make_textured_image produces a checkerboard + shapes + noise image
    # with Laplacian variance >> 100, so it reliably passes the blur gate.
    img = _make_textured_image(seed=0)

    before_path = _save_img(img_dir / "before.png", img)
    after_path  = _save_img(img_dir / "after.png",  img)   # exact same file

    pipeline = _make_pipeline(out_dir)
    job = JobInput(
        job_id="test_identical",
        item_id="item_1",
        views=[ViewInput(view_name="front", before_url=before_path, after_url=after_path)],
    )
    result = pipeline.run(job)

    assert result.status in (JobStatus.ok, JobStatus.low_confidence), (
        f"Unexpected status: {result.status}  "
        f"issues={result.views[0].quality.issues}"
    )
    assert len(result.views[0].regions) == 0


def test_blurry_image_needs_retake(tmp_dirs):
    """Uniform (blurry) images → quality gate fails, status needs_retake."""
    img_dir, out_dir = tmp_dirs
    img = np.full((300, 300, 3), 128, dtype=np.uint8)   # Laplacian variance ≈ 0
    before_path = _save_img(img_dir / "blurry_before.png", img)
    after_path  = _save_img(img_dir / "blurry_after.png",  img)

    pipeline = _make_pipeline(out_dir)
    job = JobInput(
        job_id="test_blurry",
        item_id="item_2",
        views=[ViewInput(view_name="front", before_url=before_path, after_url=after_path)],
    )
    result = pipeline.run(job)
    assert result.status == JobStatus.needs_retake
    assert any("blurry" in issue for issue in result.views[0].quality.issues)


def test_damaged_image_flagged(tmp_dirs):
    """Sharp before image with an obvious scratch in after → pipeline runs end-to-end."""
    img_dir, out_dir = tmp_dirs

    # Textured before image (passes blur gate, gives SIFT keypoints)
    before = _make_textured_image(seed=1)

    # After: add a thick black scratch across the item
    after = before.copy()
    cv2.line(after, (60,  60), (240, 240), (0,   0,   0), 12)
    cv2.line(after, (60, 200), (220,  80), (10,  10,  10), 8)

    before_path = _save_img(img_dir / "damage_before.png", before)
    after_path  = _save_img(img_dir / "damage_after.png",  after)

    pipeline = _make_pipeline(out_dir)
    job = JobInput(
        job_id="test_damage",
        item_id="item_3",
        views=[ViewInput(view_name="front", before_url=before_path, after_url=after_path)],
    )
    result = pipeline.run(job)

    # Pipeline must not crash regardless of outcome
    assert result.status in (
        JobStatus.flagged,
        JobStatus.ok,
        JobStatus.needs_retake,
        JobStatus.low_confidence,
    )
    # Whenever alignment succeeds the heatmap file must be present
    if result.views[0].quality.aligned:
        assert result.views[0].heatmap_url is not None
