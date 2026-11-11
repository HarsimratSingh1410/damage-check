"""Abstract base classes (ABCs) defining the stage interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Segmenter(ABC):
    @abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray:
        """Return binary mask (uint8, 0/255) same size as image."""


class Aligner(ABC):
    @abstractmethod
    def align(
        self, before: np.ndarray, after: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Return (aligned_after, score). Score in [0,1]. Score -1 means failure."""


class Comparator(ABC):
    @abstractmethod
    def compare(
        self, before: np.ndarray, after: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Return float32 heatmap same spatial size as inputs, values in [0,1]."""


class Reviewer(ABC):
    @abstractmethod
    def review(
        self, before_crop: np.ndarray, after_crop: np.ndarray
    ) -> tuple[bool, str]:
        """Return (is_damage, description)."""


class Storage(ABC):
    @abstractmethod
    def save_image(self, image: np.ndarray, name: str) -> str:
        """Save image, return URL/path string."""

    @abstractmethod
    def load_image(self, url: str) -> np.ndarray:
        """Load image from URL/path, return BGR uint8 ndarray."""
