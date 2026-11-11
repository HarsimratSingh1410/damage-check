"""SAM-based segmenter (Phase 3 — stub only)."""

from __future__ import annotations

import numpy as np

from ..core.interfaces import Segmenter


class SAMSegmenter(Segmenter):
    """SAM-based segmenter (Phase 3). Not yet implemented."""

    def segment(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("SAM segmenter not yet implemented.")
