"""rembg-based background segmenter."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image as PILImage

from ..core.interfaces import Segmenter


class RembgSegmenter(Segmenter):
    """Remove background with rembg and return a binary foreground mask."""

    def segment(self, image: np.ndarray) -> np.ndarray:
        from rembg import remove

        # Convert BGR → RGB PIL image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        # Encode to PNG bytes and pass to rembg
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        result_bytes = remove(buf.getvalue())

        # Decode result, extract alpha channel as binary mask
        result_pil = PILImage.open(io.BytesIO(result_bytes)).convert("RGBA")
        alpha = np.array(result_pil)[:, :, 3]
        mask = (alpha > 10).astype(np.uint8) * 255
        return mask
