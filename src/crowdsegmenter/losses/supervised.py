"""Supervised categorical cross-entropy loss for ground-truth segmentation oracle."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SupervisedCELoss(nn.Module):
    """Cross-entropy over probability predictions ``[B, K, H, W]`` and GT masks."""

    def __init__(self, ignored_value: float = 0.6, smooth: float = 1e-8) -> None:
        super().__init__()
        self.ignored_value = ignored_value
        self.smooth = smooth

    def forward(self, seg_pred: Tensor, ground_truth: Tensor) -> Tensor:
        """Compute masked cross-entropy loss between ``seg_pred`` and ``ground_truth``."""
        valid_mask = (ground_truth != self.ignored_value).all(dim=1).float()
        preds = torch.clamp(seg_pred, min=max(self.smooth, 1e-7), max=1.0)
        pixel_ce = -(ground_truth * torch.log(preds)).sum(dim=1)
        denom = torch.clamp(valid_mask.sum(), min=1.0)
        return (pixel_ce * valid_mask).sum() / denom
