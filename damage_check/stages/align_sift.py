"""SIFT + homography aligner with CLAHE brightness normalisation."""

from __future__ import annotations

import cv2
import numpy as np

from ..core.interfaces import Aligner
from ..config import settings as cfg


class SIFTAligner(Aligner):
    """Align the after-image to the before-image using SIFT keypoints and RANSAC homography."""

    def __init__(self) -> None:
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def align(
        self, before: np.ndarray, after: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, float]:
        gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

        # Detect keypoints — restrict before detection to the foreground mask
        kp1, des1 = self.sift.detectAndCompute(gray_before, mask)
        kp2, des2 = self.sift.detectAndCompute(gray_after, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return after, -1.0

        # kNN matching with Lowe's ratio test
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) < 4:
            return after, -1.0

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inlier_mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        if H is None:
            return after, -1.0

        h, w = before.shape[:2]
        aligned = cv2.warpPerspective(after, H, (w, h))

        # Score = inlier ratio
        score = (
            float(inlier_mask.sum()) / len(good)
            if inlier_mask is not None
            else 0.0
        )

        # CLAHE brightness normalisation on the L channel
        lab = cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        aligned = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return aligned, score
