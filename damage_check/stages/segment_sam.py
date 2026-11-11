"""SAM-based segmenter (Phase 3).

Segments foreground item using Segment Anything Model (SAM) from Meta.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from ..config import settings as cfg
from ..core.interfaces import Segmenter


class SAMSegmenter(Segmenter):
    """
    SAM (Segment Anything Model) segmenter.

    Uses a foundation segmentation model with central foreground prompting
    to accurately isolate the loaned item from complex backgrounds.
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        model_type: Optional[str] = None,
        device: Optional[str] = None,
        predictor: Optional[object] = None,
    ) -> None:
        self.checkpoint = checkpoint or cfg.sam_checkpoint
        self.model_type = model_type or cfg.sam_model_type
        self.device = device or cfg.device
        self._predictor = predictor

    def _get_predictor(self):
        if self._predictor is not None:
            return self._predictor

        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "segment_anything package is required for SAMSegmenter. "
                "Install it with: pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from exc

        if not self.checkpoint or not os.path.exists(self.checkpoint):
            raise FileNotFoundError(
                f"SAM checkpoint file not found: '{self.checkpoint}'. "
                "Please configure DAMAGE_SAM_CHECKPOINT in your environment."
            )

        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(device=self.device)
        self._predictor = SamPredictor(sam)
        return self._predictor

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment the foreground item from the image using SAM.

        Returns binary mask (uint8, 0 or 255) of the same spatial size as image.
        """
        predictor = self._get_predictor()
        h, w = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)

        # Center point prompt: user guided capture places item in center
        input_point = np.array([[w // 2, h // 2]], dtype=np.float32)
        input_label = np.array([1], dtype=np.int32)

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # Choose the mask with the highest confidence / IoU score
        best_idx = int(np.argmax(scores))
        mask_bool = masks[best_idx]

        mask = (mask_bool.astype(np.uint8)) * 255
        return mask
