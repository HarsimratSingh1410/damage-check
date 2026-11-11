"""DINOv2 patch-embedding comparator (Phase 3 — stub only)."""

from __future__ import annotations

import numpy as np

from ..core.interfaces import Comparator


class DINOv2Comparator(Comparator):
    """DINOv2 patch-embedding comparator (Phase 3). Not yet implemented."""

    def compare(
        self, before: np.ndarray, after: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError("DINOv2 comparator not yet implemented.")
