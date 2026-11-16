"""CT-GCE loss adapter bridging batch tensors to the verified ``CrowdSegmenter_CT_GCE`` core.

Delegates all mathematical equations directly to ``CrowdSegmenter_CT_GCE``:
    - Eq. (1): ``ctgce_loss`` (Category-Truncated Generalized Cross-Entropy)
    - Eq. (3): ``fit_dawid_skene`` and ``rho_hat`` (per-category noise estimates)
    - Eq. (4): ``calibrate_cutoffs`` (per-category truncation thresholds)
    - Eq. (7): ``c3_loss`` (coupled likelihood identifiability constraint for phi)
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from CrowdSegmenter_CT_GCE.dawid_skene import fit_dawid_skene, rho_hat as ds_rho_hat
from CrowdSegmenter_CT_GCE.identifiability import c3_loss
from CrowdSegmenter_CT_GCE.module import ctgce_loss
from CrowdSegmenter_CT_GCE.thresholds import Calibration, calibrate_cutoffs


class CTGCELoss(nn.Module):
    """PyTorch loss module wrapping the verified CT-GCE (r10) equations.

    Converts the repository's batch layout ``[B, R*K, H, W]`` (channel ``k * R + r``)
    into the verified core's tensor contract:
        - ``P``: ``[B, H, W, K]`` on the probability simplex
        - ``Y``: ``[R, B, H, W, K]`` strict one-hot targets
        - ``M``: ``[R, B, H, W]`` binary selection mask in ``{0, 1}``
        - ``Lambda``: ``[R, B, H, W, K]`` in ``[0, 1]``
    """

    def __init__(
        self,
        annotators: int = 3,
        classes: int = 2,
        ignored_value: float = 0.6,
        q: float = 0.7,
        eps: float = 1e-6,
        ablation_arm: Literal["S1", "S2", "S3"] = "S3",
    ) -> None:
        super().__init__()
        if not 0.0 < float(q) <= 1.0:
            raise ValueError(f"q must lie in (0, 1]; got {q!r}")
        if ablation_arm not in ("S1", "S2", "S3"):
            raise ValueError(f"ablation_arm must be 'S1', 'S2', or 'S3'; got {ablation_arm!r}")
        self.R = annotators
        self.K = classes
        self.ignored_value = ignored_value
        self.q = float(q)
        self.eps = eps
        self.ablation_arm = ablation_arm

    def _prepare_probabilities(self, seg_pred: Tensor) -> Tensor:
        """Convert ``[B, K, H, W]`` predictions into simplex-safe ``[B, H, W, K]``."""
        p = seg_pred.permute(0, 2, 3, 1).contiguous().float()
        p = torch.clamp(p, min=self.eps, max=1.0 - self.eps)
        return p / p.sum(dim=-1, keepdim=True)

    def unpack_annotations(self, annotations: Tensor) -> Tuple[Tensor, Tensor]:
        """Convert ``[B, R*K, H, W]`` into strict one-hot ``Y [R, B, H, W, K]`` and ``M [R, B, H, W]``."""
        b, _, h, w = annotations.shape
        ann = annotations.reshape(b, self.K, self.R, h, w).permute(2, 0, 3, 4, 1).contiguous()
        valid = (ann != self.ignored_value).all(dim=-1)  # [R, B, H, W]

        clean = torch.where(valid.unsqueeze(-1), ann, torch.zeros_like(ann))
        labels = clean.argmax(dim=-1)
        # Ensure invalid pixels still carry a valid dummy one-hot row (masked out by M=0)
        labels = torch.where(valid, labels, torch.zeros_like(labels))
        y_onehot = F.one_hot(labels, num_classes=self.K).to(dtype=torch.float32)
        m_mask = valid.to(dtype=torch.float32)
        return y_onehot, m_mask

    def warmup_loss(self, seg_pred: Tensor, annotations: Tensor) -> Tensor:
        """Plain crowd GCE (Eq. 1 with ``Lambda = 1`` and ``C = 1``) for warm-up."""
        p = self._prepare_probabilities(seg_pred)
        y, m = self.unpack_annotations(annotations)
        ones_lambda = torch.ones(
            (self.R, *p.shape), dtype=p.dtype, device=p.device
        )
        ones_c = torch.ones((self.K,), dtype=p.dtype, device=p.device)
        return ctgce_loss(p, y, m, ones_lambda, ones_c, self.q)

    def segmentation_loss(
        self,
        seg_pred: Tensor,
        lambda_pred: Tensor,
        annotations: Tensor,
        cutoffs: Tensor,
    ) -> Tensor:
        """Eq. (1): CT-GCE segmentation loss (or S1/S2 ablation weights) with ``Lambda`` detached."""
        p = self._prepare_probabilities(seg_pred)
        y, m = self.unpack_annotations(annotations)
        lam = torch.clamp(
            lambda_pred.detach().permute(1, 0, 3, 4, 2).contiguous().float(),
            min=0.0,
            max=1.0,
        )
        c = cutoffs.to(device=p.device, dtype=p.dtype)

        if self.ablation_arm != "S3":
            # Compute analytic hard truncation gate g = 1{P[k_r] > C_{k_r}} at annotated class
            p_exp = p.detach().unsqueeze(0).expand(self.R, *p.shape)
            c_exp = c.view(1, 1, 1, 1, self.K).expand_as(p_exp)
            gate = (p_exp > c_exp).to(dtype=p.dtype)
            lam = gate if self.ablation_arm == "S1" else gate * lam

        return ctgce_loss(p, y, m, lam, c, self.q)

    def reliability_loss(
        self,
        seg_pred: Tensor,
        lambda_pred: Tensor,
        annotations: Tensor,
    ) -> Tensor:
        """Eq. (7): C3 coupled likelihood for reliability branch ``phi`` with ``P`` detached."""
        p = self._prepare_probabilities(seg_pred.detach())
        y, m = self.unpack_annotations(annotations)
        lam = torch.clamp(
            lambda_pred.permute(1, 0, 3, 4, 2).contiguous().float(),
            min=0.0,
            max=1.0,
        )
        return c3_loss(p, y, m, lam)

    @torch.no_grad()
    def estimate_noise_rates(
        self,
        annotations: Tensor,
        max_iter: int = 200,
        tol: float = 1e-8,
        smoothing: float = 1e-8,
    ) -> Tensor:
        """Estimate per-category noise rates ``rho_hat [K]`` via Dawid-Skene EM (Eq. 3)."""
        y, m = self.unpack_annotations(annotations)
        labels = y.argmax(dim=-1).reshape(self.R, -1).cpu()
        mask = (m != 0).reshape(self.R, -1).cpu()
        ds = fit_dawid_skene(labels, mask, self.K, max_iter=max_iter, tol=tol, smoothing=smoothing)
        return ds_rho_hat(ds["posterior"], labels, mask, self.K)

    @torch.no_grad()
    def calibrate(
        self,
        seg_pred: Tensor,
        annotations: Tensor,
        rho_hat: Tensor,
    ) -> Calibration:
        """Calibrate Eq. (4) per-category truncation cutoffs ``C_k = max(1/K, F_k^{-1}(rho_hat_k))``."""
        p = self._prepare_probabilities(seg_pred.detach()).cpu()
        y, m = self.unpack_annotations(annotations)
        return calibrate_cutoffs(p, y.cpu(), m.cpu(), rho_hat.cpu())

    def forward(
        self,
        seg_pred: Tensor,
        ann_pred: Optional[Tensor],
        annotations: Tensor,
        anns_ids: Optional[Tensor] = None,
        cutoffs: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute CT-GCE segmentation loss (falls back to warm-up GCE when ``cutoffs`` is None)."""
        if cutoffs is None or ann_pred is None:
            return self.warmup_loss(seg_pred, annotations)
        return self.segmentation_loss(seg_pred, ann_pred, annotations, cutoffs)
