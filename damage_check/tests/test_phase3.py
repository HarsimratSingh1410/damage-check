"""Tests for Phase 3 advanced stages: DINOv2 comparator, SAM segmenter, and LoFTR aligner."""

from __future__ import annotations

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch

from damage_check.stages.align_loftr import LoFTRAligner
from damage_check.stages.compare_dino import DINOv2Comparator
from damage_check.stages.segment_sam import SAMSegmenter


# ---------------------------------------------------------------------------
# DINOv2 Comparator Tests
# ---------------------------------------------------------------------------

def test_dinov2_identical_images():
    """Identical feature embeddings should yield a zero-distance heatmap."""
    H, W = 140, 140
    num_patches = (H // 14) * (W // 14)
    embed_dim = 64

    # Fixed mock feature representation
    fixed_features = torch.randn(1, num_patches, embed_dim)

    mock_model = MagicMock()
    mock_model.get_intermediate_layers.return_value = [fixed_features]

    comparator = DINOv2Comparator(model=mock_model)

    before = np.full((H, W, 3), 128, dtype=np.uint8)
    after = before.copy()
    mask = np.full((H, W), 255, dtype=np.uint8)

    heatmap = comparator.compare(before, after, mask)

    assert heatmap.shape == (H, W)
    assert heatmap.dtype == np.float32
    # Since features were identical, cosine similarity is 1.0 -> distance is 0.0
    np.testing.assert_allclose(heatmap, 0.0, atol=1e-5)


def test_dinov2_different_images():
    """Different feature embeddings should produce high distance in the heatmap."""
    H, W = 140, 140
    num_patches = (H // 14) * (W // 14)
    embed_dim = 64

    feats1 = torch.ones(1, num_patches, embed_dim)
    # Orthogonal features -> cosine similarity 0.0 -> distance 1.0
    feats2 = -torch.ones(1, num_patches, embed_dim)

    mock_model = MagicMock()
    mock_model.get_intermediate_layers.side_effect = [[feats1], [feats2]]

    comparator = DINOv2Comparator(model=mock_model)

    before = np.full((H, W, 3), 100, dtype=np.uint8)
    after = np.full((H, W, 3), 200, dtype=np.uint8)
    mask = np.full((H, W), 255, dtype=np.uint8)

    heatmap = comparator.compare(before, after, mask)
    assert heatmap.shape == (H, W)
    assert heatmap.max() > 0.5


def test_dinov2_masking():
    """Background pixels defined by mask=0 should be zeroed in the heatmap."""
    H, W = 140, 140
    num_patches = (H // 14) * (W // 14)
    embed_dim = 32

    feats1 = torch.randn(1, num_patches, embed_dim)
    feats2 = torch.randn(1, num_patches, embed_dim)

    mock_model = MagicMock()
    mock_model.get_intermediate_layers.side_effect = [[feats1], [feats2]]

    comparator = DINOv2Comparator(model=mock_model)

    before = np.zeros((H, W, 3), dtype=np.uint8)
    after = np.ones((H, W, 3), dtype=np.uint8) * 255

    # Center circle foreground mask
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask, (70, 70), 30, 255, -1)

    heatmap = comparator.compare(before, after, mask)

    # Background must be zero
    assert np.all(heatmap[mask == 0] == 0.0)


# ---------------------------------------------------------------------------
# SAM Segmenter Tests
# ---------------------------------------------------------------------------

def test_sam_segment_with_mock_predictor():
    """SAMSegmenter should correctly process predictor masks and return binary uint8 mask."""
    H, W = 200, 200
    mock_predictor = MagicMock()

    # Return a simulated boolean mask with score 0.95
    simulated_mask = np.zeros((H, W), dtype=bool)
    simulated_mask[50:150, 50:150] = True
    mock_predictor.predict.return_value = (
        np.array([simulated_mask]),
        np.array([0.95]),
        None,
    )

    segmenter = SAMSegmenter(predictor=mock_predictor)
    img = np.full((H, W, 3), 128, dtype=np.uint8)

    mask = segmenter.segment(img)

    mock_predictor.set_image.assert_called_once()
    mock_predictor.predict.assert_called_once()

    assert mask.shape == (H, W)
    assert mask.dtype == np.uint8
    assert np.all(mask[50:150, 50:150] == 255)
    assert np.all(mask[0:40, 0:40] == 0)


def test_sam_missing_checkpoint():
    """If checkpoint file does not exist, SAMSegmenter should raise FileNotFoundError."""
    segmenter = SAMSegmenter(checkpoint="nonexistent_checkpoint.pth")
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    # When segment_anything is not installed or file is missing
    with pytest.raises((ImportError, FileNotFoundError)):
        segmenter.segment(img)


# ---------------------------------------------------------------------------
# LoFTR Aligner Tests
# ---------------------------------------------------------------------------

def test_loftr_aligner_with_mock_matcher():
    """LoFTRAligner should compute homography from matched keypoints and warp the image."""
    H, W = 200, 200
    mock_matcher = MagicMock()

    # 4 perfect corner matches (identity transform)
    pts0 = torch.tensor([[20.0, 20.0], [180.0, 20.0], [180.0, 180.0], [20.0, 180.0]])
    pts1 = pts0.clone()
    conf = torch.tensor([0.9, 0.95, 0.92, 0.88])

    mock_matcher.return_value = {
        "keypoints0": pts0,
        "keypoints1": pts1,
        "confidence": conf,
    }

    aligner = LoFTRAligner(matcher=mock_matcher)

    before = np.full((H, W, 3), 100, dtype=np.uint8)
    after = np.full((H, W, 3), 100, dtype=np.uint8)

    aligned, score = aligner.align(before, after)

    assert aligned.shape == (H, W, 3)
    assert score > 0.5


def test_loftr_aligner_insufficient_matches():
    """LoFTRAligner should return -1.0 score when fewer than 4 matches are found."""
    mock_matcher = MagicMock()
    mock_matcher.return_value = {
        "keypoints0": torch.tensor([[10.0, 10.0]]),
        "keypoints1": torch.tensor([[10.0, 10.0]]),
        "confidence": torch.tensor([0.9]),
    }

    aligner = LoFTRAligner(matcher=mock_matcher)
    before = np.zeros((100, 100, 3), dtype=np.uint8)
    after = np.zeros((100, 100, 3), dtype=np.uint8)

    aligned, score = aligner.align(before, after)
    assert score == -1.0


# ---------------------------------------------------------------------------
# End-to-End Pipeline Integration with Phase 3 Stages
# ---------------------------------------------------------------------------

def test_pipeline_integration_with_dinov2(tmp_path):
    """Pipeline should execute cleanly when configured with DINOv2 comparator."""
    from damage_check.core.models import JobInput, ViewInput, JobStatus
    from damage_check.core.pipeline import Pipeline
    from damage_check.stages.align_sift import SIFTAligner
    from damage_check.storage.local import LocalStorage
    from damage_check.tests.test_pipeline import DummySegmenter, _make_textured_image, _save_img

    out_dir = tmp_path / "out"
    img_dir = tmp_path / "images"
    out_dir.mkdir()
    img_dir.mkdir()

    img = _make_textured_image(seed=99)
    before_path = _save_img(img_dir / "before.png", img)
    after_path = _save_img(img_dir / "after.png", img)

    # Mock DINOv2 returning zero change (identical features)
    num_patches = (img.shape[0] // 14) * (img.shape[1] // 14)
    fixed_feats = torch.randn(1, num_patches, 32)
    mock_model = MagicMock()
    mock_model.get_intermediate_layers.return_value = [fixed_feats]

    pipeline = Pipeline(
        segmenter=DummySegmenter(),
        aligner=SIFTAligner(),
        comparator=DINOv2Comparator(model=mock_model),
        storage=LocalStorage(output_dir=str(out_dir)),
    )

    job = JobInput(
        job_id="dinov2-e2e-job",
        item_id="item-dino",
        views=[ViewInput(view_name="front", before_url=before_path, after_url=after_path)],
    )

    result = pipeline.run(job)
    assert result.status in (JobStatus.ok, JobStatus.low_confidence)
    assert len(result.views) == 1
    assert result.views[0].quality.aligned is True
    assert result.views[0].heatmap_url is not None

