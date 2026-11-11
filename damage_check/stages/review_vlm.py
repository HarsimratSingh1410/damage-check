"""VLM-based reviewer (Phase 4 — stub only)."""

from __future__ import annotations

import numpy as np

from ..core.interfaces import Reviewer


class VLMReviewer(Reviewer):
    """VLM-based reviewer (Phase 4). Not yet implemented."""

    def review(
        self, before_crop: np.ndarray, after_crop: np.ndarray
    ) -> tuple[bool, str]:
        raise NotImplementedError("VLM reviewer not yet implemented.")
