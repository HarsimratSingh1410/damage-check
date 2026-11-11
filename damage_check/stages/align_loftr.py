"""LoFTR-based aligner (Phase 3 — stub only)."""

from __future__ import annotations

import numpy as np

from ..core.interfaces import Aligner


class LoFTRAligner(Aligner):
    """LoFTR-based aligner (Phase 3). Not yet implemented."""

    def align(
        self, before: np.ndarray, after: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, float]:
        raise NotImplementedError("LoFTR aligner not yet implemented.")
