"""Non-parametric baseline models for multi-annotator segmentation.

Provides:
    1. MajorityVoting: pixel-wise soft vote across valid annotators.
    2. SpatialSTAPLE: vectorized multi-class STAPLE with a Potts MRF spatial prior
       (beta = 0.0 reduces to classical non-spatial STAPLE).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8
LOG_EPS = 1e-12


class BaselineModel(nn.Module):
    """Marker base class for non-parametric baseline models."""

    is_baseline: bool = True


class MajorityVoting(BaselineModel):
    """Parameter-free PyTorch module computing pixel-wise majority vote across annotators.

    Excludes sentinel pixels (equal to ``ignored_value``).
    """

    def __init__(
        self,
        num_annotators: int,
        num_classes: int,
        ignored_value: float = 0.6,
    ) -> None:
        super().__init__()
        self.num_annotators = num_annotators
        self.num_classes = num_classes
        self.ignored_value = ignored_value

    def forward(
        self,
        masks: torch.Tensor,
        anns_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """Compute pixel-wise majority vote across valid annotators.

        Args:
            masks: Annotator stack of shape ``[B, A * C, H, W]`` with channel
                layout ``k * A + r``.
            anns_ids: Optional presence vector ``[B, A]`` (unused; validity is
                determined pixel-wise via ``ignored_value``).

        Returns:
            Tuple ``(soft_votes, None)`` where ``soft_votes`` has shape
            ``[B, C, H, W]`` in ``[0, 1]``.
        """
        b, _, h, w = masks.shape
        masks_reshaped = masks.reshape(
            b, self.num_classes, self.num_annotators, h, w
        ).transpose(1, 2)  # [B, A, C, H, W]

        valid_mask = masks_reshaped != self.ignored_value
        masks_clean = torch.where(valid_mask, masks_reshaped, torch.zeros_like(masks_reshaped))
        votes_sum = masks_clean.sum(dim=1)
        votes_count = valid_mask.float().sum(dim=1)

        soft_votes = torch.where(
            votes_count > 0,
            votes_sum / votes_count,
            torch.zeros_like(votes_sum),
        )
        return soft_votes, None


# ---------------------------------------------------------------------------
# Vectorized Multi-Class Spatial STAPLE (Potts / MRF prior)
# ---------------------------------------------------------------------------

def build_potts_kernel(
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build depthwise 3x3 Potts kernel: center 0, 8 neighbours 1.

    Returns:
        Weight tensor of shape ``[K, 1, 3, 3]`` for grouped ``conv2d``.
    """
    w = torch.ones(num_classes, 1, 3, 3, dtype=dtype, device=device)
    w[:, 0, 1, 1] = 0.0
    return w


def initialize_parameters(
    D: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Initialize annotator confusion matrices and global class prior.

    Args:
        D: ``[R, K, H, W]`` one-hot annotator votes (zeroed where missing).
        valid_mask: Optional boolean mask ``[R, H, W]`` indicating annotated pixels.

    Returns:
        ``Theta`` of shape ``[R, K, K]`` (0.85 diag / 0.15 off-diag) and
        ``gamma`` of shape ``[K]`` empirical class prior over valid votes.
    """
    if D.ndim != 4:
        raise ValueError(f"D must be 4D (R, K, H, W), got {tuple(D.shape)}")
    r_count, k_count = D.shape[0], D.shape[1]
    off_diag = 0.15 / max(k_count - 1, 1)
    theta = torch.full((r_count, k_count, k_count), off_diag, dtype=D.dtype, device=D.device)
    idx = torch.arange(k_count, device=D.device)
    theta[:, idx, idx] = 0.85

    if valid_mask is not None:
        valid_count = valid_mask.to(torch.float64).sum().clamp(min=1.0)
        gamma = (D.to(torch.float64).sum(dim=(0, 2, 3)) / valid_count).to(D.dtype)
    else:
        gamma = D.to(torch.float64).mean(dim=(0, 2, 3)).to(D.dtype)

    gamma = torch.clamp(gamma, min=EPS)
    gamma = gamma / gamma.sum()
    return theta, gamma


def e_step(
    D: torch.Tensor,
    theta: torch.Tensor,
    gamma: torch.Tensor,
    beta: float,
    W_prev: Optional[torch.Tensor],
    potts_weight: torch.Tensor,
) -> torch.Tensor:
    """E-step: posterior latent map ``W`` of shape ``[K, H, W]``."""
    log_theta = torch.log(torch.clamp(theta, min=LOG_EPS))
    ll = torch.einsum("rlhw,rkl->khw", D, log_theta)
    log_gamma = torch.log(torch.clamp(gamma, min=LOG_EPS)).view(-1, 1, 1)
    f_score = ll + log_gamma
    if W_prev is not None and beta != 0.0:
        neigh = F.conv2d(W_prev.unsqueeze(0), potts_weight, padding=1, groups=W_prev.shape[0])
        f_score = f_score + beta * neigh.squeeze(0)
    f_score = f_score - f_score.max(dim=0, keepdim=True).values
    exp_f = torch.exp(f_score)
    return exp_f / torch.clamp(exp_f.sum(dim=0, keepdim=True), min=EPS)


def m_step(D: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """M-step: row-stochastic confusion matrices ``Theta`` of shape ``[R, K, K]``."""
    numerator = torch.einsum("khw,rlhw->rkl", W, D)
    # Only count pixels where each annotator actually provided a vote
    ann_present = D.sum(dim=1)  # [R, H, W], 1 where valid, 0 where missing
    denom = torch.einsum("khw,rhw->rk", W, ann_present).unsqueeze(-1)  # [R, K, 1]
    k_count = W.shape[0]
    theta = (numerator + EPS) / (denom + k_count * EPS)
    return theta / torch.clamp(theta.sum(dim=-1, keepdim=True), min=EPS)


def compute_entropy(W: torch.Tensor) -> torch.Tensor:
    """Pixel-wise entropy ``H = -sum_k W_k log W_k`` of shape ``[H, W]``."""
    wc = torch.clamp(W, min=LOG_EPS, max=1.0)
    return -(wc * torch.log(wc)).sum(dim=0)


def run_spatial_staple(
    D: torch.Tensor,
    beta: float = 1.0,
    max_iter: int = 20,
    tol: float = 1e-4,
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Run EM loop for multi-class Spatial STAPLE on a single ``[R, K, H, W]`` sample.

    Iteration 0 uses likelihood + prior only; iterations ``1..max_iter`` include
    the Potts spatial prior ``beta * conv(W_prev)``.
    """
    D = D.float()
    with torch.no_grad():
        _, k_count, _, _ = D.shape
        theta, gamma = initialize_parameters(D, valid_mask=valid_mask)
        potts_weight = build_potts_kernel(k_count, device=D.device, dtype=D.dtype)

        W_history: List[torch.Tensor] = []
        Theta_history: List[torch.Tensor] = []
        deltas: List[float] = []

        W_prev = e_step(D, theta, gamma, beta=0.0, W_prev=None, potts_weight=potts_weight)
        W_history.append(W_prev.clone())
        Theta_history.append(theta.clone())
        deltas.append(0.0)

        converged = False
        n_iter = 0
        for t in range(1, max_iter + 1):
            W_new = e_step(D, theta, gamma, beta=beta, W_prev=W_prev, potts_weight=potts_weight)
            theta = m_step(D, W_new)
            delta = float((W_new - W_prev).abs().mean().item())
            W_history.append(W_new.clone())
            Theta_history.append(theta.clone())
            deltas.append(delta)
            W_prev = W_new
            n_iter = t
            if delta < tol:
                converged = True
                break

        return {
            "W": W_prev,
            "Theta": theta,
            "W_history": W_history,
            "Theta_history": Theta_history,
            "deltas": deltas,
            "gamma": gamma,
            "n_iter": n_iter,
            "converged": converged,
        }


class SpatialSTAPLE(BaselineModel):
    """Vectorized PyTorch Multi-Class Spatial STAPLE baseline with Potts MRF prior.

    Accepts the standard ``CrowdSegmenter`` annotator stack ``[B, A * C, H, W]``
    and returns consensus class probabilities ``[B, C, H, W]`` (plus ``None``
    for the auxiliary head slot). Set ``beta=0.0`` for classical non-spatial STAPLE.
    """

    def __init__(
        self,
        num_annotators: int,
        num_classes: int,
        ignored_value: float = 0.6,
        beta: float = 1.0,
        max_iter: int = 20,
        tol: float = 1e-4,
    ) -> None:
        super().__init__()
        self.num_annotators = num_annotators
        self.num_classes = num_classes
        self.ignored_value = ignored_value
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol

    def _to_onehot_votes(self, sample_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert ``[A, C, H, W]`` mask values into strict one-hot ``D`` and ``valid_mask``."""
        valid = (sample_masks != self.ignored_value).all(dim=1)  # [A, H, W]
        clean = torch.where(valid.unsqueeze(1), sample_masks, torch.zeros_like(sample_masks))
        ordinal = clean.argmax(dim=1)  # [A, H, W]
        onehot = F.one_hot(ordinal, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        onehot = onehot * valid.unsqueeze(1).float()
        return onehot, valid

    def forward(
        self,
        masks: torch.Tensor,
        anns_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """Run Spatial STAPLE EM per sample in the batch.

        Args:
            masks: Annotator stack ``[B, A * C, H, W]`` with layout ``k * A + r``.
            anns_ids: Optional annotator presence vector ``[B, A]`` (unused).

        Returns:
            Tuple ``(consensus_probs, None)`` with shape ``[B, C, H, W]``.
        """
        b, _, h, w = masks.shape
        masks_reshaped = masks.reshape(
            b, self.num_classes, self.num_annotators, h, w
        ).transpose(1, 2)  # [B, A, C, H, W]

        outputs: List[torch.Tensor] = []
        for i in range(b):
            d_i, valid_i = self._to_onehot_votes(masks_reshaped[i])
            if not bool(valid_i.any()):
                uniform = torch.full(
                    (self.num_classes, h, w),
                    1.0 / self.num_classes,
                    dtype=torch.float32,
                    device=masks.device,
                )
                outputs.append(uniform)
                continue
            res = run_spatial_staple(
                d_i,
                beta=self.beta,
                max_iter=self.max_iter,
                tol=self.tol,
                valid_mask=valid_i,
            )
            outputs.append(res["W"])  # type: ignore[arg-type]

        return torch.stack(outputs, dim=0), None
