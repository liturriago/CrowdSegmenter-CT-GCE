"""Noisy Label Loss for multi-annotator segmentation via learned confusion matrices."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class NoisyLabelLoss(nn.Module):
    """Noisy Label Loss for crowdsourcing segmentation via learned confusion matrices.

    Models annotator reliability by routing predictions through a per-pixel
    confusion matrix before computing negative log-likelihood, with optional
    trace regularization.
    """

    def __init__(
        self,
        annotators: int,
        classes: int,
        ignored_value: float = 0.6,
        min_trace: bool = False,
        alpha: float = 0.1,
        smooth: float = 1e-8,
    ) -> None:
        super().__init__()
        self.annotators = annotators
        self.classes = classes
        self.ignored_value = ignored_value
        self.min_trace = min_trace
        self.alpha = alpha
        self.smooth = smooth
        self.nll_loss = nn.NLLLoss(reduction="mean", ignore_index=-100)

    def _select_annotator_masks(
        self,
        annotations: Tensor,
        anns_ids: Tensor,
    ) -> Tensor:
        """Select the ``[B, C, H, W]`` channels matching the one-hot ``anns_ids``."""
        b, _, h, w = annotations.shape
        r = anns_ids.argmax(dim=1)
        k = torch.arange(self.classes, device=annotations.device)
        indices = k.unsqueeze(0) * self.annotators + r.unsqueeze(1)
        indices = indices.unsqueeze(-1).unsqueeze(-1).expand(b, self.classes, h, w)
        return annotations.gather(dim=1, index=indices)

    def forward(
        self,
        pred: Tensor,
        cms: Tensor,
        annotations: Tensor,
        anns_ids: Tensor,
    ) -> Tensor:
        """Compute Noisy Label Loss for the selected annotator."""
        b, c, h, w = pred.size()

        labels = self._select_annotator_masks(annotations, anns_ids)
        valid_mask = (labels != self.ignored_value).all(dim=1)

        labels_int = labels.argmax(dim=1).long()
        labels_int[~valid_mask] = -100

        pred_flat = (
            pred.view(b, c, h * w)
            .permute(0, 2, 1)
            .contiguous()
            .view(b * h * w, c, 1)
        )

        cm = (
            cms.view(b, c ** 2, h * w)
            .permute(0, 2, 1)
            .contiguous()
            .view(b * h * w, c, c)
        )
        cm = cm / (cm.sum(dim=1, keepdim=True) + self.smooth)

        pred_noisy = torch.bmm(cm, pred_flat)
        pred_noisy = (
            pred_noisy.view(b, h * w, c)
            .permute(0, 2, 1)
            .contiguous()
            .view(b, c, h, w)
        )

        loss_ce = self.nll_loss(torch.log(pred_noisy + self.smooth), labels_int)
        regularisation = (
            self.alpha * torch.trace(cm.sum(dim=0).T).sum() / (b * h * w)
        )
        return loss_ce + regularisation if self.min_trace else loss_ce - regularisation
