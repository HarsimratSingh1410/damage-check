"""Pipeline orchestrator for the Damage-Check Module."""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

from ..core.models import (
    DamageRegion,
    JobInput,
    JobResult,
    JobStatus,
    QualityInfo,
    ViewInput,
    ViewResult,
)
from ..core.interfaces import Aligner, Comparator, Reviewer, Segmenter, Storage
from ..config import settings as cfg


class Pipeline:
    def __init__(
        self,
        segmenter: Segmenter,
        aligner: Aligner,
        comparator: Comparator,
        storage: Storage,
        reviewer: Optional[Reviewer] = None,
    ):
        self.segmenter = segmenter
        self.aligner = aligner
        self.comparator = comparator
        self.storage = storage
        self.reviewer = reviewer

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, job: JobInput) -> JobResult:
        view_results = []
        for view in job.views:
            vr = self._process_view(view, job.job_id)
            view_results.append(vr)
        overall = self._aggregate_status(view_results)
        return JobResult(job_id=job.job_id, status=overall, views=view_results)

    # ------------------------------------------------------------------
    # Per-view processing
    # ------------------------------------------------------------------

    def _process_view(self, view: ViewInput, job_id: str) -> ViewResult:
        before = self.storage.load_image(view.before_url)
        after = self.storage.load_image(view.after_url)

        # 1. Quality gate
        issues = self._quality_check(before, after)
        if issues:
            return ViewResult(
                view_name=view.view_name,
                quality=QualityInfo(aligned=False, alignment_score=0.0, issues=issues),
                regions=[],
            )

        # 2. Segment
        mask = self.segmenter.segment(before)

        # 3. Align
        aligned_after, align_score = self.aligner.align(before, after, mask)
        if align_score < cfg.alignment_min_score:
            return ViewResult(
                view_name=view.view_name,
                quality=QualityInfo(
                    aligned=False,
                    alignment_score=float(align_score),
                    issues=["alignment_failed"],
                ),
                regions=[],
            )

        # 4. Compare → heatmap
        heatmap = self.comparator.compare(before, aligned_after, mask)

        # 5. Extract candidate regions
        regions = self._extract_regions(
            heatmap, before, aligned_after, view.view_name, job_id
        )

        # 6. Optional VLM review
        if self.reviewer and regions:
            regions = self._vlm_filter(regions, before, aligned_after)

        # 7. Save heatmap visualization
        heatmap_vis = (heatmap * 255).clip(0, 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
        heatmap_url = self.storage.save_image(
            heatmap_color, f"{job_id}_{view.view_name}_heatmap"
        )

        return ViewResult(
            view_name=view.view_name,
            quality=QualityInfo(
                aligned=True, alignment_score=float(align_score), issues=[]
            ),
            regions=regions,
            heatmap_url=heatmap_url,
        )

    # ------------------------------------------------------------------
    # Quality gate
    # ------------------------------------------------------------------

    def _quality_check(self, before: np.ndarray, after: np.ndarray) -> list:
        issues = []
        for name, img in [("before", before), ("after", after)]:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < cfg.blur_threshold:
                issues.append(f"{name}_blurry")
            mean = float(gray.mean())
            if mean < cfg.brightness_min:
                issues.append(f"{name}_too_dark")
            elif mean > cfg.brightness_max:
                issues.append(f"{name}_too_bright")
        return issues

    # ------------------------------------------------------------------
    # Region extraction
    # ------------------------------------------------------------------

    def _extract_regions(
        self,
        heatmap: np.ndarray,
        before: np.ndarray,
        after: np.ndarray,
        view_name: str,
        job_id: str,
    ) -> list:
        # Threshold the heatmap
        binary = (heatmap > cfg.damage_score_threshold).astype(np.uint8) * 255

        # Morphological cleanup: close small gaps, remove isolated pixels
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        h_img, w_img = before.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < cfg.min_region_pixels:
                continue

            x, y, w, wh = cv2.boundingRect(cnt)
            # Clamp to image bounds
            x, y = max(0, x), max(0, y)
            w = min(w, w_img - x)
            wh = min(wh, h_img - y)
            if w <= 0 or wh <= 0:
                continue

            # Score = mean heatmap value inside the contour
            region_mask = np.zeros(heatmap.shape, dtype=np.uint8)
            cv2.drawContours(region_mask, [cnt], -1, 1, -1)
            pixels = heatmap[region_mask == 1]
            score = float(pixels.mean()) if pixels.size > 0 else 0.0

            # Save crops
            crop_b = before[y : y + wh, x : x + w]
            crop_a = after[y : y + wh, x : x + w]
            crop_b_url = self.storage.save_image(
                crop_b, f"{job_id}_{view_name}_region_{x}_{y}_before"
            )
            crop_a_url = self.storage.save_image(
                crop_a, f"{job_id}_{view_name}_region_{x}_{y}_after"
            )

            regions.append(
                DamageRegion(
                    bbox=[x, y, w, wh],
                    score=round(score, 4),
                    crop_before_url=crop_b_url,
                    crop_after_url=crop_a_url,
                )
            )

        # Sort by score descending
        regions.sort(key=lambda r: r.score, reverse=True)
        return regions

    # ------------------------------------------------------------------
    # VLM filter
    # ------------------------------------------------------------------

    def _vlm_filter(self, regions, before, after):
        kept = []
        for r in regions:
            x, y, w, wh = r.bbox
            crop_b = before[y : y + wh, x : x + w]
            crop_a = after[y : y + wh, x : x + w]
            is_damage, label = self.reviewer.review(crop_b, crop_a)
            if is_damage:
                r.vlm_label = label
                kept.append(r)
        return kept

    # ------------------------------------------------------------------
    # Status aggregation
    # ------------------------------------------------------------------

    def _aggregate_status(self, view_results: list) -> JobStatus:
        all_unaligned = all(not vr.quality.aligned for vr in view_results)
        any_unaligned = any(not vr.quality.aligned for vr in view_results)

        if all_unaligned:
            return JobStatus.needs_retake

        if any(vr.regions for vr in view_results):
            return JobStatus.flagged

        # If some views failed alignment but we have results from others, still ok
        return JobStatus.ok
