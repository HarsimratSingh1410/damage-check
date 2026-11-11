"""LoFTR-based aligner (Phase 3).

Detector-Free Local Feature Matching with Transformers for robust alignment
across wide baselines and low-texture surfaces.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from ..config import settings as cfg
from ..core.interfaces import Aligner


class LoFTRAligner(Aligner):
    """
    LoFTR (Local Feature TRansformer) aligner.

    Uses dense transformer-based matching (via Kornia) to find reliable correspondences
    even on smooth surfaces or difficult lighting where SIFT yields too few keypoints.
    """

    def __init__(
        self,
        pretrained: Optional[str] = None,
        device: Optional[str] = None,
        matcher: Optional[object] = None,
        confidence_thresh: float = 0.4,
    ) -> None:
        self.pretrained = pretrained or cfg.loftr_pretrained
        self.device = device or cfg.device
        self.confidence_thresh = confidence_thresh
        self._matcher = matcher

    def _get_matcher(self):
        if self._matcher is not None:
            return self._matcher

        try:
            import torch
            import kornia.feature as KF
        except ImportError as exc:
            raise ImportError(
                "torch and kornia are required for LoFTRAligner. "
                "Install them with: pip install torch kornia"
            ) from exc

        device_obj = torch.device(self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu")
        matcher = KF.LoFTR(pretrained=self.pretrained)
        matcher.eval()
        matcher.to(device_obj)
        self._matcher = matcher
        return self._matcher

    def align(
        self, before: np.ndarray, after: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Align 'after' image to 'before' image using dense transformer matches.

        Returns (aligned_after, alignment_score). Score in [0, 1]. Returns -1.0 on failure.
        """
        import torch

        matcher = self._get_matcher()
        h, w = before.shape[:2]

        # Convert to grayscale
        gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

        # LoFTR expects grayscale float tensors [1, 1, H, W] in [0, 1]
        t0 = torch.from_numpy(gray_before).float().unsqueeze(0).unsqueeze(0) / 255.0
        t1 = torch.from_numpy(gray_after).float().unsqueeze(0).unsqueeze(0) / 255.0

        device = next(matcher.parameters()).device if hasattr(matcher, "parameters") and list(matcher.parameters()) else torch.device("cpu")
        t0 = t0.to(device)
        t1 = t1.to(device)

        input_dict = {"image0": t0, "image1": t1}
        with torch.no_grad():
            output = matcher(input_dict)

        pts0 = output["keypoints0"].cpu().numpy()
        pts1 = output["keypoints1"].cpu().numpy()
        confidence = output.get("confidence", None)
        if confidence is not None:
            confidence = confidence.cpu().numpy()

        if len(pts0) < 4:
            return after, -1.0

        # Filter by confidence threshold
        if confidence is not None:
            good_idx = confidence >= self.confidence_thresh
            pts0 = pts0[good_idx]
            pts1 = pts1[good_idx]

        # Filter by foreground mask if available
        if mask is not None and len(pts0) > 0:
            valid_pts = []
            for i, p in enumerate(pts0):
                px, py = int(round(p[0])), int(round(p[1]))
                if 0 <= py < h and 0 <= px < w and mask[py, px] > 0:
                    valid_pts.append(i)
            if len(valid_pts) >= 4:
                pts0 = pts0[valid_pts]
                pts1 = pts1[valid_pts]

        if len(pts0) < 4:
            return after, -1.0

        # RANSAC homography from 'after' (pts1) to 'before' (pts0)
        H, inlier_mask = cv2.findHomography(pts1, pts0, cv2.RANSAC, 5.0)
        if H is None or inlier_mask is None:
            return after, -1.0

        inliers = inlier_mask.ravel() == 1
        inlier_count = int(np.sum(inliers))
        if inlier_count < 4:
            return after, -1.0

        score = float(inlier_count) / float(len(pts0))

        # Warp 'after' image into before's frame of reference
        aligned = cv2.warpPerspective(after, H, (w, h))

        # Color/brightness normalization: CLAHE on L channel of LAB
        lab = cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        aligned = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return aligned, score
