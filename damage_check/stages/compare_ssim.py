"""SSIM-based comparator producing a per-pixel damage heatmap."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from ..core.interfaces import Comparator


class SSIMComparator(Comparator):
    """Compare before/after using SSIM and return an inverted difference map."""

    def compare(
        self, before: np.ndarray, after: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        # Convert to grayscale
        gray_b = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        gray_a = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

        # SSIM with full=True returns the per-pixel similarity map
        _score, diff = ssim(gray_b, gray_a, full=True, data_range=255)

        # diff ∈ [-1, 1]; low similarity → high damage score
        damage_map = 1.0 - ((diff + 1.0) / 2.0)
        damage_map = damage_map.astype(np.float32)

        # Zero out background pixels
        mask_f = (mask > 0).astype(np.float32)
        damage_map *= mask_f

        return damage_map
