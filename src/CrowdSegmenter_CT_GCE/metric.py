"""The evaluation metric: Neighborhood Latent Consistency, Eqs. (8) and (9).

r09 § "The evaluation metric: neighborhood latent consistency" defines, with no
ground truth anywhere:

    p(x, Y) = norm[ norm(MAP(F(x), Y)), norm(MAP(F(x), B_Y)) ],              (8)

    C_a = 1/(|N_a| (|N_a|-1)) * sum_{i != j in N_a} p(x_i, Y_a^i)^T p(x_j, Y_a^j),  (9)

with `B_Y = dilate(Y) - erode(Y)` the boundary ring, `F(x)` the dense features
of patch x "spatially centered and L2-normalized per pixel", `N_a` the set of
neighborhood patches annotator `a` drew and `Y_a^i` their mask on patch `i`.
A panel score summarizes {C_a}; the revision names the candidates: the mean,
the median, the lower quantile, and the fraction of annotators below a
threshold. Which summary and which normalization of `C_a` are selected is the
pilot's question (`metric_pilot.py`); nothing here answers it.

Port note. The formulation is ported from the first prototype
(`/home/lucas/repos/new_metric`, outside this repository) rather than
rewritten: the prototype measured the PANEL-CONSENSUS variant of Eq. (8)/(9)
(one mask per patch, the panel's consensus) and observed the compressed ladder
its README records (0.7984 -> 0.7865 over contamination c = 0..10). Two ported
choices are declared rather than re-derived:

- the feature preparation chain is the prototype's: per-channel spatial mean
  subtraction, per-pixel L2 normalization, then a second spatial centering —
  the "mean-leakage elimination" premise of Eq. (8); the second centering
  leaves each pixel vector with non-unit norm, which is harmless because both
  prototypes of Eq. (8) are normalized afterwards;
- the morphological operators are the prototype's 7x7 square window (radius 3),
  with the prototype's zero boundary behaviour: a window crossing the image
  border sees zeros, so the eroded mask is 0 on any such window.

Divergence from the prototype, declared: Eq. (9) divides by
|N_a| (|N_a| - 1), which is undefined for |N_a| < 2; the prototype returned
1.0 there. This module fails closed instead — a single-patch neighborhood
raises.

One module per mathematical object; `__provenance__` is read statically by
`implementation_cli.py verify` — the module is never imported to obtain it —
and it is the only thing that makes revision drift detectable.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.ndimage import maximum_filter

__provenance__ = {
    "revision": "research-concept-category-truncated-generalized-cross-entropy-for-semantic-se-r10.md",
    "sections": ["The evaluation metric: neighborhood latent consistency"],
    "equations": ["8", "9"],
    "invariants": [
        "metric_prototype_unit_norm",
        "metric_boundary_ring_dilate_minus_erode",
        "metric_centering_kills_offset",
        "metric_consistency_matches_pairwise_average",
        "metric_consistency_bounds",
        "metric_consistency_fails_closed_single_patch",
        "metric_panel_summaries_match_manual",
    ],
}

#: The prototype's numerical slack: added to every mask-sum denominator before
#: the division that forms a masked average (MAP), and to every norm.
EPS = 1e-8

#: The morphological radius of the boundary ring: dilate/erode run over a
#: (2 * RING_RADIUS + 1)^2 window, the prototype's 7x7 square.
RING_RADIUS = 3


def prepare_features(F: np.ndarray) -> np.ndarray:
    """The feature premise of Eq. (8): spatially centered, L2-normalized.

    `F` is a dense feature map [C, H, W] for ONE patch (the caller batches).
    The ported chain, in the prototype's order: subtract the per-channel
    spatial mean, L2-normalize each pixel vector, then subtract the per-channel
    spatial mean again. The second centering is the prototype's "zero-mean
    spatial orthogonalization"; both centerings together are what make a
    constant per-channel offset vanish (invariant
    `metric_centering_kills_offset`).
    """
    F = np.asarray(F, dtype=np.float32)
    if F.ndim != 3:
        raise ValueError(f"F must be [C, H, W]; got shape {F.shape}")
    if not np.isfinite(F).all():
        raise ValueError("F must be finite")
    F = F - F.mean(axis=(1, 2), keepdims=True)
    norms = np.linalg.norm(F, axis=0, keepdims=True)
    F = F / (norms + EPS)
    F = F - F.mean(axis=(1, 2), keepdims=True)
    return F


def boundary_ring(Y: np.ndarray, radius: int = RING_RADIUS) -> np.ndarray:
    """The boundary ring of Eq. (8): B_Y = dilate(Y) - erode(Y).

    `Y` is a mask [H, W] with values in [0, 1] (binary or soft). Dilation is a
    maximum filter and erosion the negative of a maximum filter over the
    negated mask, both over a (2*radius+1)^2 square window, both with the
    prototype's zero padding: a window crossing the border sees zeros, so the
    erosion is 0 on any window that touches the border. The result is clipped
    to [0, 1], as the prototype does.
    """
    Y = np.asarray(Y, dtype=np.float32)
    if Y.ndim != 2:
        raise ValueError(f"Y must be [H, W]; got shape {Y.shape}")
    if not np.isfinite(Y).all():
        raise ValueError("Y must be finite")
    size = 2 * int(radius) + 1
    dilated = maximum_filter(Y, size=size, mode="constant", cval=0.0)
    eroded = -maximum_filter(-Y, size=size, mode="constant", cval=0.0)
    return np.clip(dilated - eroded, 0.0, 1.0)


def _unit(v: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D vector, with the prototype's epsilon slack."""
    n = float(np.linalg.norm(v))
    return v / (n + EPS)


def prototype(F: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Eq. (8): the dual-component latent prototype of mask Y on patch x.

    `F` is the PREPARED dense feature map [C, H, W] of the patch (run
    `prepare_features` first — the extraction side lives in the benchmark
    package's material, not here). `Y` is the annotator's mask [H, W],
    resampled by the caller to F's own grid (the pilot resamples with the
    prototype's bilinear resize; the equation only consumes aligned arrays).

    Foreground: p_fg = norm(MAP(F, Y)); boundary: p_bnd = norm(MAP(F, B_Y));
    output: p = norm([p_fg, p_bnd]) in R^{2C}. Every MAP is a masked average
    with the prototype's epsilon in the denominator. An empty mask maps to the
    zero vector (its masked sum is empty), the prototype's own behaviour.
    """
    F = np.asarray(F, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    if F.ndim != 3:
        raise ValueError(f"F must be [C, H, W]; got shape {F.shape}")
    if Y.shape != F.shape[1:]:
        raise ValueError(f"Y must be [H, W] = {F.shape[1:]}; got shape {Y.shape}")
    if not np.isfinite(Y).all():
        raise ValueError("Y must be finite")

    B = boundary_ring(Y)
    fg_sum = Y.sum() + EPS
    bnd_sum = B.sum() + EPS
    # MAP: masked average over pixels, one value per channel.
    p_fg = _unit(np.einsum("chw,hw->c", F, Y, optimize=True) / fg_sum)
    p_bnd = _unit(np.einsum("chw,hw->c", F, B, optimize=True) / bnd_sum)
    return _unit(np.concatenate([p_fg, p_bnd]))


def consistency(prototypes: Sequence[np.ndarray]) -> tuple[float, np.ndarray]:
    """Eq. (9): the neighborhood latent consistency of ONE annotator.

    `C_a` is the mean pairwise cosine similarity of the annotator's own
    prototypes across the neighborhood: the mean of the off-diagonal entries
    of the prototype Gram matrix, with the divisor |N_a| (|N_a| - 1) of the
    equation. Returns (C_a, gram) where gram is the full similarity matrix —
    the pilot reuses it; the score itself is only the off-diagonal mean.

    Fails closed for |N_a| < 2: the divisor is zero and the equation is
    undefined. The prototype returned 1.0 there; that leniency is NOT ported
    (see the module docstring).
    """
    P = np.stack([np.asarray(p, dtype=np.float64) for p in prototypes], axis=0)
    if P.ndim != 2:
        raise ValueError(f"prototypes must stack to [N, D]; got shape {P.shape}")
    n = P.shape[0]
    if n < 2:
        raise ValueError(
            f"Eq. (9) needs |N_a| >= 2 neighborhood patches; got {n}. "
            "The divisor |N_a| (|N_a|-1) is zero below 2 and the prototype's "
            "1.0 fallback is not ported.")
    if not np.isfinite(P).all():
        raise ValueError("prototypes must be finite")
    gram = P @ P.T
    off = gram.sum() - np.trace(gram)
    c = float(off / (n * (n - 1)))
    return c, gram


def panel_mean(scores: Sequence[float]) -> float:
    """Panel summary 1: the arithmetic mean of the annotators' scores."""
    values = _panel_values(scores)
    return float(np.mean(values))


def panel_median(scores: Sequence[float]) -> float:
    """Panel summary 2: the median of the annotators' scores."""
    values = _panel_values(scores)
    return float(np.median(values))


def panel_lower_quantile(scores: Sequence[float], q: float = 0.1) -> float:
    """Panel summary 3: the lower quantile of the annotators' scores.

    `q` in [0, 1) is the quantile of the SCORE distribution, so q = 0.1 is the
    lower decile: the pilot's declared value, sensitive to the panel's worst
    annotators. Linear interpolation between order statistics (numpy's
    default), declared.
    """
    if not 0.0 <= q < 1.0:
        raise ValueError(f"q must lie in [0, 1); got {q!r}")
    values = _panel_values(scores)
    return float(np.quantile(values, q))


def calibration_threshold(population: Sequence[float], k: float = 1.0) -> float:
    """The fraction-below threshold rule: mu - k * sigma of a calibration population.

    The pilot's tau is one population standard deviation below the mean of the
    calibration cohort (the expert pool), computed in whatever scale the
    variant produces — for z-standardized variants this is exactly z = -k.
    Population sigma (ddof = 0), declared. Fails closed on an empty
    population or a non-positive k.
    """
    values = _panel_values(population)
    if k <= 0:
        raise ValueError(f"k must be positive; got {k!r}")
    return float(np.mean(values) - k * np.std(values))


def fraction_below(scores: Sequence[float], tau: float) -> float:
    """Panel summary 4: the fraction of annotators below the threshold tau.

    Direction: this summary GROWS with contamination (more annotators below
    the expert rule), so its decision direction is opposite to the other
    three; the pilot declares `direction` per row and never mixes them.
    """
    values = _panel_values(scores)
    return float(np.mean(np.asarray(values) < tau))


def _panel_values(scores: Sequence[float]) -> np.ndarray:
    """Validate and stack a panel's scores: finite, non-empty, 1-D."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"panel scores must be a non-empty 1-D sequence; got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("panel scores must be finite")
    return values
