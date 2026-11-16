"""TGCE-SSPS loss (Truncated Generalized Cross-Entropy for Semantic Segmentation)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class TGCE_SSPS(nn.Module):
    """Truncated Generalized Cross-Entropy for Semantic Segmentation Pixel Selection."""

    def __init__(
        self,
        annotators: int = 3,
        classes: int = 2,
        ignored_value: float = 0.6,
        q: float = 0.48029,
        lambda_factor: float = 1.0,
        smooth: float = 1e-7,
    ) -> None:
        super().__init__()
        self.K = classes
        self.R = annotators
        self.ignored_value = ignored_value
        self.q = q
        self.smooth = smooth
        self.lambda_factor = lambda_factor

    def forward(
        self,
        seg_pred: Tensor,
        ann_pred: Tensor,
        annotations: Tensor,
        anns_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute TGCE-SSPS loss over ``seg_pred [B, K, H, W]`` and ``ann_pred [B, R, H, W]``."""
        device = seg_pred.device
        epsilon = 1e-8

        y_pred_classes = torch.clamp(seg_pred, epsilon, 1.0 - epsilon)
        lambda_r = ann_pred

        n, _, h, w = y_pred_classes.shape
        ann = annotations.reshape(n, self.K, self.R, h, w).contiguous()

        valid_mask = torch.all(ann != self.ignored_value, dim=1).float()
        ann = torch.where(ann != self.ignored_value, ann, torch.zeros_like(ann))

        y_pred_expanded = y_pred_classes.unsqueeze(2).repeat(1, 1, self.R, 1, 1)

        term_r = torch.mean(
            ann * (1 - torch.pow(y_pred_expanded, self.q)) / (self.q + epsilon + self.smooth),
            dim=1,
        )

        uniform_probs = (1 / self.K + self.smooth) * torch.ones(
            (n, self.R, h, w), device=device
        )
        term_c = (1 - lambda_r) * (1 - torch.pow(uniform_probs, self.q)) / (
            self.q + epsilon + self.smooth
        )

        combined_terms = lambda_r * term_r + term_c
        masked_terms = combined_terms * valid_mask
        valid_pixel_count = torch.maximum(
            torch.sum(valid_mask), torch.tensor(1.0, device=device)
        )
        return torch.sum(masked_terms) / valid_pixel_count
