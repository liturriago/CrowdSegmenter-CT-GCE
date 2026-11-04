"""The per-category truncation thresholds of Eq. (4), and their calibration.

r03 § "Estimating the truncation thresholds" calibrates the threshold of
category k as the rho_hat_k-quantile of the model's confidence on the pixels
labeled k, clipped at 1/K:

    C_k = max{ 1/K, F_k^{-1}(rho_hat_k) },   F_k(t) = P(P[w,h,k] <= t | k_r(w,h) = k).  (4)

The conditioning of F_k is essential: `category_confidences` returns, per
category, only the confidences P[pixel, k] of the annotations (r, pixel) that
carry M = 1 and were labeled k — never all pixels.

Empirical generalized inverse (documented convention): sort the n values;
index = ceil(level * n) - 1, clamped to [0, n-1]; level = 0 returns the minimum
(the clamp of index -1 does that on its own). The floor is 1/K, so the result
lies in [1/K, 1].

Diagnostics returned alongside the cutoffs: the raw `quantile` per category,
whether it was `clipped` below the floor (quantile < 1/K), and the honest
`discarded` fraction mean(values_k <= C_k) = F_k(C_k) — the fraction of the
annotations of category k the objective actually drops, which the floor can
push above rho_hat_k.

A category with no annotated pixel is a precondition failure (ValueError): no
threshold is ever invented. A NaN rho_hat_k (the Dawid-Skene estimate has no
support for that category) is refused the same way.

One module per mathematical object; `__provenance__` is read statically by
`implementation_cli.py verify` — the module is never imported to obtain it —
and it is the only thing that makes revision drift detectable.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from .reliability import annotated_labels

__provenance__ = {
    "revision": "research-concept-category-truncated-generalized-cross-entropy-for-semantic-se-r09.md",
    "sections": ["Estimating the truncation thresholds"],
    "equations": ["4"],
    "invariants": [
        "cutoff_at_least_inverse_k",
        "cutoff_within_unit_interval",
        "cutoff_monotone_in_rho",
        "quantile_matches_empirical_cdf",
        "category_confidences_conditioned_on_annotated_label",
        "calibration_requires_support",
        "discarded_fraction_is_at_least_rho",
    ],
}


class Calibration(NamedTuple):
    """What Eq. (4)'s calibration produces for every category.

    cutoffs:  C_k = max(1/K, quantile_k), in [1/K, 1];
    quantile: the raw rho_hat_k-quantile F_k^{-1}(rho_hat_k), before the floor;
    clipped:  quantile_k < 1/K, i.e. the 1/K floor actually bound;
    discarded: mean(values_k <= C_k) = F_k(C_k), the fraction of category k's
               annotations the truncation drops.
    """

    cutoffs: torch.Tensor
    quantile: torch.Tensor
    clipped: torch.Tensor
    discarded: torch.Tensor


def category_confidences(P: torch.Tensor, Y: torch.Tensor,
                         M: torch.Tensor) -> list:
    """F_k's sample, one 1-D tensor per category.

    Entry set for category k: `P[pixel, k]` for every ANNOTATION (r, pixel)
    with M[r, pixel] = 1 and annotated label k. The conditioning on
    `k_r(w,h) = k` of Eq. (4) is what this enforces — pixels labeled j != k are
    excluded from category k's list even when P[pixel, k] is large, and pixels
    nobody annotated (M = 0) are excluded outright.
    """
    if not isinstance(P, torch.Tensor) or P.ndim < 2:
        raise ValueError(f"P must be a tensor of shape [..., K]; got {P!r}")
    labels = annotated_labels(Y)
    if P.shape[:-1] != labels.shape[1:]:
        raise ValueError(
            f"P must have shape [*S, K] with labels [R, *S]; got P {tuple(P.shape)}, "
            f"labels {tuple(labels.shape)}")
    if M.shape != labels.shape:
        raise ValueError(f"M must have shape {tuple(labels.shape)}; got {tuple(M.shape)}")
    if not bool(((M == 0) | (M == 1)).all()):
        raise ValueError("M must be a selection mask with values exactly 0 or 1")
    n_classes = P.shape[-1]
    P_exp = P.unsqueeze(0).expand(labels.shape[0], *P.shape)
    selected = M != 0
    out = []
    for k in range(n_classes):
        pick = selected & (labels == k)
        out.append(P_exp[..., k][pick].reshape(-1).to(torch.float64))
    return out


def empirical_quantile(values: torch.Tensor, level: float) -> torch.Tensor:
    """The empirical generalized inverse F_n^{-1}(level), as a scalar tensor.

    Convention: sort the values; index = ceil(level * n) - 1 clamped to
    [0, n - 1]; level = 0 therefore returns the minimum and level = 1 the
    maximum. Raises ValueError on empty `values` or a `level` outside [0, 1].
    """
    if not isinstance(values, torch.Tensor) or values.ndim != 1:
        raise ValueError("values must be a 1-D tensor")
    if values.numel() == 0:
        raise ValueError("values must be non-empty")
    level = float(level)
    if not 0 <= level <= 1:
        raise ValueError(f"level must lie in [0, 1]; got {level!r}")
    n = values.numel()
    index = math.ceil(level * n) - 1
    index = min(max(index, 0), n - 1)
    return values.sort().values[index].reshape(())


def calibrate_cutoffs(P: torch.Tensor, Y: torch.Tensor, M: torch.Tensor,
                      rho_hat: torch.Tensor) -> Calibration:
    """Eq. (4): C_k = max(1/K, F_k^{-1}(rho_hat_k)), plus its diagnostics.

    `rho_hat` is the [K] vector of estimated per-category noise rates (r03 Eq.
    (3) estimated through `dawid_skene.rho_hat`); a NaN entry is refused — the
    estimate has no support for that category, and no threshold is invented for
    it. A category whose sample F_k is empty (no annotated pixel carries that
    label) is refused the same way.
    """
    confidences = category_confidences(P, Y, M)
    n_classes = len(confidences)
    if not isinstance(rho_hat, torch.Tensor) or rho_hat.shape != (n_classes,):
        raise ValueError(f"rho_hat must have shape ({n_classes},); got {rho_hat!r}")
    rho_hat = rho_hat.to(torch.float64)
    if not bool(torch.isfinite(rho_hat).all()):
        raise ValueError("rho_hat must be finite: a NaN noise estimate has no "
                         "support and its threshold cannot be calibrated")
    if bool((rho_hat < 0).any()) or bool((rho_hat > 1).any()):
        raise ValueError("rho_hat must lie in [0, 1]")

    quantile = torch.empty(n_classes, dtype=torch.float64)
    for k in range(n_classes):
        values = confidences[k]
        if values.numel() == 0:
            raise ValueError(
                f"category {k} has no annotated pixel: its confidence sample is "
                "empty and Eq. (4) has no quantile to take")
        quantile[k] = empirical_quantile(values, float(rho_hat[k]))

    floor = 1.0 / n_classes
    cutoffs = torch.clamp(quantile, min=floor)
    clipped = quantile < floor
    discarded = torch.empty(n_classes, dtype=torch.float64)
    for k in range(n_classes):
        values = confidences[k]
        discarded[k] = (values <= cutoffs[k]).to(torch.float64).mean()
    return Calibration(cutoffs=cutoffs, quantile=quantile, clipped=clipped,
                       discarded=discarded)
