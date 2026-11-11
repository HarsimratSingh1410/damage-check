"""DINOv2 patch-embedding comparator (Phase 3).

Compares before and after images by computing the distance between
dense patch embeddings extracted from a foundation vision transformer (DINOv2).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..config import settings as cfg
from ..core.interfaces import Comparator


class DINOv2Comparator(Comparator):
    """
    Comparator based on DINOv2 patch tokens.

    Extracts dense feature representations for patches of both images and
    computes cosine distance to create a robust semantic change heatmap.
    """

    PATCH_SIZE = 14
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        model: Optional[object] = None,
    ) -> None:
        self.model_name = model_name or cfg.dino_model
        self.device = device or cfg.device
        self._model = model

    def _get_model(self):
        if self._model is not None:
            return self._model

        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required for DINOv2Comparator.") from exc

        device_obj = torch.device(self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu")
        try:
            model = torch.hub.load("facebookresearch/dinov2", self.model_name)
        except Exception as err:
            raise RuntimeError(
                f"Failed to load DINOv2 model '{self.model_name}' from torch.hub: {err}"
            ) from err

        model.eval()
        model.to(device_obj)
        self._model = model
        return self._model

    def _preprocess(self, img_bgr: np.ndarray, target_size: tuple[int, int]):
        """Convert BGR uint8 to normalized PyTorch tensor [1, 3, H, W]."""
        import torch

        # Convert to RGB and resize to target patch-multiple dimensions
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

        # Normalize with ImageNet mean/std
        norm = (resized.astype(np.float32) / 255.0 - self.IMAGENET_MEAN) / self.IMAGENET_STD
        # Transpose from HWC to CHW
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor

    def compare(
        self, before: np.ndarray, after: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Compare before and aligned after image using DINOv2 patch embeddings.

        Returns float32 heatmap with values in [0.0, 1.0], masked to foreground.
        """
        import torch
        import torch.nn.functional as F

        model = self._get_model()
        orig_h, orig_w = before.shape[:2]

        # Calculate dimensions that are divisible by PATCH_SIZE (14)
        target_h = max(self.PATCH_SIZE, (orig_h // self.PATCH_SIZE) * self.PATCH_SIZE)
        target_w = max(self.PATCH_SIZE, (orig_w // self.PATCH_SIZE) * self.PATCH_SIZE)

        # Clamp to reasonable size for inference (e.g. max 518px)
        max_dim = 518
        if max(target_h, target_w) > max_dim:
            scale = max_dim / max(target_h, target_w)
            target_h = max(self.PATCH_SIZE, int((target_h * scale) // self.PATCH_SIZE) * self.PATCH_SIZE)
            target_w = max(self.PATCH_SIZE, int((target_w * scale) // self.PATCH_SIZE) * self.PATCH_SIZE)

        grid_h = target_h // self.PATCH_SIZE
        grid_w = target_w // self.PATCH_SIZE

        t_before = self._preprocess(before, (target_h, target_w))
        t_after = self._preprocess(after, (target_h, target_w))

        device = next(model.parameters()).device if hasattr(model, "parameters") and list(model.parameters()) else torch.device("cpu")
        t_before = t_before.to(device)
        t_after = t_after.to(device)

        with torch.no_grad():
            if hasattr(model, "get_intermediate_layers"):
                # Standard DINOv2 architecture
                feats_before = model.get_intermediate_layers(t_before, n=1)[0]
                feats_after = model.get_intermediate_layers(t_after, n=1)[0]
            elif callable(model):
                # Generic model callable that outputs [B, num_patches, D]
                feats_before = model(t_before)
                feats_after = model(t_after)
            else:
                raise ValueError("Model does not support feature extraction.")

            # Cosine similarity between corresponding patches
            sim = F.cosine_similarity(feats_before, feats_after, dim=-1)  # [1, num_patches]
            sim = sim.view(1, 1, grid_h, grid_w)

            # Cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite
            # Damage is high distance
            dist = 1.0 - sim
            dist = torch.clamp(dist, 0.0, 1.0)

            # Interpolate distance map back to original image resolution
            heatmap_tensor = F.interpolate(dist, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            heatmap = heatmap_tensor.squeeze().cpu().numpy().astype(np.float32)

        # Apply foreground mask
        if mask is not None:
            mask_f = (mask > 0).astype(np.float32)
            heatmap *= mask_f

        # Zero out negligible numerical noise and clamp to [0.0, 1.0]
        heatmap[heatmap < 1e-4] = 0.0
        heatmap = np.clip(heatmap, 0.0, 1.0)

        return heatmap.astype(np.float32)
