"""Population pieces for the bound study — the pieces, never a bound.

r03 § "Falsifiable claim and assumptions" states the bound as a HYPOTHESIS to
be tested ("This is the hypothesis to be tested, not a result"), under a
crowd-noise model "with per-annotator, per-category flip rates
rho_{r,k -> k'}, conditionally independent given the consensus segmentation".
This module implements the population objects such a bound would be stated
over, and nothing more:

- `CrowdModel`: the noise model itself — prior [K], confusions [R, K, K] with
  T_r[k,k'] = P(observed k' | truth k), rows summing to one — validated;
- `noise_rates`: the per-annotator per-category flip rates (off-diagonal mass)
  and diagonal correctness;
- `population_rho`: the population counterpart of Eq. (3), averaging over
  annotators uniformly (the loss's E_r): rho_{r,k} = 1 - prior[k]*T_r[k,k] /
  sum_{k'} prior[k']*T_r[k',k], then rho_k = mean_r rho_{r,k};
- `consensus_posterior`: P(k* = k | annotations) under conditional independence;
- `bayes_risk`: the exact expected 0-1 risk of the Bayes predictor given all R
  annotations, by enumerating all K^R observation vectors (R <= 4; it refuses
  larger R and says so). In [0, 1];
- `expected_truncated_loss`: E over truth and annotations of
  min((1 - Q[y]^q)/q, (1 - C[y]^q)/q) — Eq. (1) at the pointwise optimal Lambda
  of Eq. (2), averaged over annotators, where y is the annotated category
  k_r(w,h), exactly the index Eq. (1) gathers at. This isolates the truncation;
- `compare_cutoffs`: both expected losses and their difference, for a
  per-category cutoff vector and a single global cutoff. It reports the
  measured difference and never asserts the per-category vector is tighter —
  r03's primary claim says the bound with the vector is tighter, and that claim
  is a hypothesis this module does not implement.

One module per mathematical object; `__provenance__` is read statically by
`implementation_cli.py verify` — the module is never imported to obtain it —
and it is the only thing that makes revision drift detectable.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch

__provenance__ = {
    "revision": "research-concept-category-truncated-generalized-cross-entropy-for-semantic-se-r09.md",
    "sections": ["Falsifiable claim and assumptions", "Estimating the truncation thresholds"],
    "equations": ["1", "2", "3"],
    "invariants": [
        "crowd_model_rows_normalized",
        "population_rho_matches_monte_carlo",
        "bayes_risk_matches_monte_carlo",
        "bayes_risk_is_bounded",
        "global_cutoff_is_nested",
        "consensus_posterior_without_annotations_is_prior",
    ],
}

#: Slack for the simplex checks (a row computed in float64 can land an ulp off).
SIMPLEX_TOL = 1e-9
#: `bayes_risk` enumerates all K^R observation vectors; above this R it refuses.
BAYES_ENUMERATION_MAX_ANNOTATORS = 4


@dataclass
class CrowdModel:
    """The per-annotator per-category flip model of r03's primary claim.

    prior [K]: P(k* = k), sums to one. confusions [R, K, K]:
    T_r[k,k'] = P(observed k' | truth k), every row sums to one. Both are
    validated on construction (ValueError otherwise).
    """

    prior: torch.Tensor
    confusions: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.prior, torch.Tensor) or self.prior.ndim != 1:
            raise ValueError(f"prior must be a 1-D tensor [K]; got {self.prior!r}")
        k = self.prior.numel()
        if k < 1:
            raise ValueError("prior must carry at least one category")
        if not bool(torch.isfinite(self.prior).all()):
            raise ValueError("prior must be finite")
        if bool((self.prior < -SIMPLEX_TOL).any()):
            raise ValueError("prior must lie in [0, 1]")
        if abs(float(self.prior.sum()) - 1.0) > SIMPLEX_TOL:
            raise ValueError("prior must sum to 1")
        confusions = self.confusions
        if not isinstance(confusions, torch.Tensor) or confusions.ndim != 3:
            raise ValueError(
                f"confusions must be [R, K, K]; got {confusions!r}")
        if confusions.shape[1:] != (k, k):
            raise ValueError(
                f"confusions must be [R, {k}, {k}] to match the prior; "
                f"got {tuple(confusions.shape)}")
        if not bool(torch.isfinite(confusions).all()):
            raise ValueError("confusions must be finite")
        if bool((confusions < -SIMPLEX_TOL).any()) or bool((confusions > 1 + SIMPLEX_TOL).any()):
            raise ValueError("confusions must lie in [0, 1]")
        row_sums = confusions.sum(dim=-1)
        if not bool((row_sums - 1).abs().max() <= SIMPLEX_TOL):
            raise ValueError("every confusion row must sum to 1")
        self.prior = self.prior.to(torch.float64)
        self.confusions = self.confusions.to(torch.float64)

    @property
    def n_classes(self) -> int:
        return self.prior.numel()

    @property
    def n_annotators(self) -> int:
        return self.confusions.shape[0]


def noise_rates(model: CrowdModel) -> dict:
    """Per-annotator per-category flip rates and diagonal correctness.

    {"flip": [R, K], "correct": [R, K]}: flip[r,k] = sum_{k' != k} T_r[k,k'],
    the probability that annotator r moves a true k elsewhere; correct[r,k] =
    T_r[k,k], the probability they keep it. flip + correct = 1 by construction.
    """
    T = model.confusions
    correct = T.diagonal(dim1=-2, dim2=-1).clone()
    flip = 1.0 - correct
    return {"flip": flip, "correct": correct}


def population_rho(model: CrowdModel) -> torch.Tensor:
    """The population counterpart of Eq. (3), per category, in [0, 1] or NaN.

    For annotator r: rho_{r,k} = P(k* != k | k_r = k) = 1 - prior[k]*T_r[k,k] /
    sum_{k'} prior[k']*T_r[k',k], exactly Eq. (3) with the model's joint. r03's
    loss averages over annotators uniformly (its E_r), so rho_k = mean_r
    rho_{r,k}. A category no observation probability mass reaches has an
    undefined conditioning and is NaN.
    """
    prior, T = model.prior, model.confusions
    k = model.n_classes
    out = torch.zeros(k, dtype=torch.float64)
    for r in range(model.n_annotators):
        # obs_mass[k] = P(k_r = k) = sum_{k'} prior[k'] * T_r[k', k]
        obs_mass = prior @ T[r]                       # [K]
        joint_diag = prior * T[r].diagonal()           # [K]
        with torch.no_grad():
            rho_r = torch.where(
                obs_mass > 0,
                1.0 - joint_diag / obs_mass,
                torch.full_like(obs_mass, float("nan")))
        out = out + rho_r
    return out / model.n_annotators


def consensus_posterior(model: CrowdModel, labels: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """P(k* = k | annotations), [N, K], under conditional independence (A1).

    log posterior[n,k] = log prior[k] + sum_r mask[r,n] * log T_r[k, labels[r,n]],
    normalized with logsumexp. labels [R, N] long, mask [R, N] bool.
    """
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ValueError(f"labels must be [R, N]; got {labels!r}")
    if labels.shape[0] != model.n_annotators:
        raise ValueError(
            f"labels must carry {model.n_annotators} annotators; got {labels.shape[0]}")
    if mask.shape != labels.shape:
        raise ValueError(f"mask must have shape {tuple(labels.shape)}; got {mask!r}")
    if not bool(((labels >= 0) & (labels < model.n_classes)).all()):
        raise ValueError(f"labels must lie in [0, {model.n_classes})")
    sel = mask != 0
    log_unnorm = torch.log(model.prior.clamp_min(1e-300)).unsqueeze(0).repeat(
        labels.shape[1], 1)
    for r in range(model.n_annotators):
        cell = model.confusions[r][:, labels[r].to(torch.int64)].T  # [N, K]
        log_unnorm = log_unnorm + sel[r].to(torch.float64).unsqueeze(-1) * torch.log(
            cell.clamp_min(1e-300))
    return torch.softmax(log_unnorm, dim=1)


def bayes_risk(model: CrowdModel) -> float:
    """The exact expected 0-1 risk of the Bayes predictor given all R annotations.

    Enumerates every observation vector o in {1..K}^R (so it refuses R > 4 and
    says so: the enumeration is K^R terms). For each o the joint is
    P(y, o) = prior[y] * prod_r T_r[y, o_r], and the Bayes error contributed by
    o is P(o) - max_y P(y, o); the risk is their sum over o, equal to
    1 - sum_o max_y P(y, o). Lies in [0, 1] by construction.
    """
    if model.n_annotators > BAYES_ENUMERATION_MAX_ANNOTATORS:
        raise ValueError(
            f"bayes_risk enumerates all K^R observation vectors and refuses "
            f"R > {BAYES_ENUMERATION_MAX_ANNOTATORS}; got R = {model.n_annotators}")
    prior, T = model.prior, model.confusions
    k = model.n_classes
    best_total = 0.0
    for obs in itertools.product(range(k), repeat=model.n_annotators):
        joint = prior.clone()
        for r, o_r in enumerate(obs):
            joint = joint * T[r][:, o_r]
        best_total += float(joint.max())
    risk = 1.0 - best_total
    return min(max(risk, 0.0), 1.0)


def expected_truncated_loss(model: CrowdModel, predictor: torch.Tensor,
                            cutoffs: torch.Tensor, q: float) -> float:
    """Eq. (1) at the pointwise optimal Lambda of Eq. (2), averaged over annotators.

    At the optimum the integrand is min((1 - Q[y]^q)/q, (1 - C[y]^q)/q) with y
    the annotated category k_r(w,h) — exactly the index Eq. (1) gathers at. The
    expectation is over truth and annotations: sum_r (1/R) sum_y prior[y]
    sum_{y'} T_r[y, y'] * min((1 - Q[y']^q)/q, (1 - C[y']^q)/q). This isolates
    the truncation: no reliability branch, no training, no bound.
    """
    if predictor.shape != (model.n_classes,):
        raise ValueError(
            f"predictor must have shape ({model.n_classes},); got {tuple(predictor.shape)}")
    if not bool(torch.isfinite(predictor).all()):
        raise ValueError("predictor must be finite")
    if bool((predictor < -SIMPLEX_TOL).any()) or bool((predictor > 1 + SIMPLEX_TOL).any()):
        raise ValueError("predictor must lie in [0, 1]")
    if abs(float(predictor.sum()) - 1.0) > SIMPLEX_TOL:
        raise ValueError("predictor must sum to 1")
    if cutoffs.shape != (model.n_classes,):
        raise ValueError(
            f"cutoffs must have shape ({model.n_classes},); got {tuple(cutoffs.shape)}")
    if bool((cutoffs <= 0).any()) or bool((cutoffs > 1).any()):
        raise ValueError("cutoffs must lie in (0, 1]")
    q = float(q)
    if not 0 < q <= 1:
        raise ValueError(f"q must lie in (0, 1]; got {q!r}")
    Q = predictor.to(torch.float64)
    C = cutoffs.to(torch.float64)
    terms = torch.minimum((1 - Q ** q) / q, (1 - C ** q) / q)   # [K], indexed by y'
    total = 0.0
    for r in range(model.n_annotators):
        obs_mass = model.prior @ model.confusions[r]             # [K], P(k_r = y')
        total += float((obs_mass * terms).sum())
    return total / model.n_annotators


def compare_cutoffs(model: CrowdModel, predictor: torch.Tensor,
                    per_category: torch.Tensor, global_cutoff: float,
                    q: float) -> dict:
    """Both expected truncated losses and their difference; nothing asserted.

    Returns {"per_category", "global", "difference"} with difference =
    per_category - global (negative means the per-category cutoffs yield the
    smaller expected truncated loss). When `per_category` is constant and equals
    `global_cutoff` the difference is exactly zero: a single global cutoff is
    the nested special case of the vector. The per-category vector is never
    asserted tighter here — r03 states that as the hypothesis of its primary
    claim, and this reports the measured difference only.
    """
    per = expected_truncated_loss(model, predictor, per_category, q)
    global_vec = torch.full_like(per_category.to(torch.float64), float(global_cutoff))
    glob = expected_truncated_loss(model, predictor, global_vec, q)
    return {"per_category": per, "global": glob, "difference": per - glob}
