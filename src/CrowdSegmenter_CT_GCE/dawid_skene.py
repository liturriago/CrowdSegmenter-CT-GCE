"""Dawid-Skene EM over the crowd masks: Eq. (3)'s estimator.

r03 § "Estimating the truncation thresholds" defines

    rho_k = P( k_r(w,h) != k*(w,h) | k_r(w,h) = k )                         (3)

and states it "is estimated primarily by a Dawid-Skene-style
expectation-maximization over the crowd masks, which yields per-annotator
confusion matrices T_r[k,k'] = P(k_r = k' | k* = k) and hence rho_k under
conditional independence of annotators given the consensus".

`fit_dawid_skene` is that EM on the [R, N] annotation grid (R annotators, N
pixels): labels[r, n] is the annotated category and mask[r, n] says whether
annotator r drew pixel n.

Estimator details (not changes to r03): the posterior is INITIALIZED from the
per-pixel majority vote — ties take the lowest class index, a pixel with no
annotation gets a uniform row — to anchor the label permutation every latent-
class EM carries. The E-step runs in log space (log prior + masked sum of log
T_r[k, labels], normalized with logsumexp); the M-step sets prior = posterior
mean and T_r[k,k'] proportional to the posterior mass of the pixels annotator r
labeled k', smoothed and row-normalized; the log-likelihood is
sum_n logsumexp(log_post unnormalized) and the run stops when |delta| < tol.

`rho_hat` averages (1 - posterior[n,k]) over every annotation (r,n) that is
masked and labeled k — Eq. (3)'s conditioning on the ANNOTATED category. A
category no annotation carries has no estimate and is NaN, never invented; the
threshold calibration refuses it there.

`majority_disagreement` is the model-free cross-check r03 names ("the
pixel-wise disagreement rate against the majority of annotators serves as a
model-free cross-check that needs no ground truth"). It is NOT rho_k and is not
presented as an estimate of it: it conditions on the majority vote, not on the
consensus k*.

One module per mathematical object; `__provenance__` is read statically by
`implementation_cli.py verify` — the module is never imported to obtain it —
and it is the only thing that makes revision drift detectable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__provenance__ = {
    "revision": "research-concept-category-truncated-generalized-cross-entropy-for-semantic-se-r10.md",
    "sections": ["Estimating the truncation thresholds"],
    "equations": ["3"],
    "invariants": [
        "posterior_normalized",
        "confusion_rows_normalized",
        "em_log_likelihood_nondecreasing",
        "rho_recovers_synthetic_noise",
        "majority_disagreement_matches_manual_count",
    ],
}

#: Floor on the entries of T_r before taking logs in the E-step, so a zero cell
#: (possible only when smoothing = 0) cannot poison the posterior with -inf.
_LOG_FLOOR = 1e-300


def _validated(labels: torch.Tensor, mask: torch.Tensor, n_classes: int):
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ValueError(f"labels must be a [R, N] tensor; got {labels!r}")
    if not isinstance(mask, torch.Tensor) or mask.shape != labels.shape:
        raise ValueError(f"mask must have shape {tuple(labels.shape)}; got {mask!r}")
    if n_classes < 1:
        raise ValueError("n_classes must be at least 1")
    if not bool(((labels >= 0) & (labels < n_classes)).all()):
        raise ValueError(f"labels must lie in [0, {n_classes})")
    if labels.dtype not in (torch.int32, torch.int64, torch.int16, torch.int8):
        raise ValueError("labels must be an integer tensor")
    return mask != 0


def majority_posterior(labels: torch.Tensor, mask: torch.Tensor,
                       n_classes: int) -> torch.Tensor:
    """The majority-vote initialization: one-hot argmax per pixel, [N, K].

    Ties take the LOWEST class index (torch.argmax's own tie rule), and a pixel
    no annotator drew gets a uniform row. This anchors the label permutation of
    the EM — an estimator detail, documented in the module docstring.
    """
    sel = _validated(labels, mask, n_classes)
    n = labels.shape[1]
    counts = torch.zeros(n, n_classes, dtype=torch.float64)
    for r in range(labels.shape[0]):
        counts.scatter_add_(1, labels[r].to(torch.int64).unsqueeze(-1),
                            sel[r].to(torch.float64).unsqueeze(-1))
    posterior = torch.zeros(n, n_classes, dtype=torch.float64)
    has = counts.sum(dim=1) > 0
    if bool(has.any()):
        rows = has.nonzero(as_tuple=False).squeeze(1)
        winner = counts[has].argmax(dim=1)
        posterior[rows, winner] = 1.0
    if bool((~has).any()):
        posterior[~has] = 1.0 / n_classes
    return posterior


def _m_step(posterior: torch.Tensor, labels: torch.Tensor, sel: torch.Tensor,
            n_classes: int, smoothing: float):
    """prior = posterior.mean(0); T_r[k,k'] = smoothed, row-normalized
    posterior mass of the pixels annotator r labeled k' among those whose
    posterior sits on k."""
    prior = posterior.mean(dim=0)
    r_annotators = labels.shape[0]
    confusions = torch.zeros(r_annotators, n_classes, n_classes, dtype=torch.float64)
    for r in range(r_annotators):
        weights = posterior * sel[r].to(torch.float64).unsqueeze(-1)  # [N, K]
        one_hot = F.one_hot(labels[r].to(torch.int64), n_classes).to(torch.float64)
        numer = weights.T @ one_hot                                    # [K, K']
        denom = weights.sum(dim=0)                                     # [K]
        confusions[r] = ((numer + smoothing)
                         / (denom.unsqueeze(-1) + n_classes * smoothing))
    return prior, confusions


def _e_step(prior: torch.Tensor, confusions: torch.Tensor,
            labels: torch.Tensor, sel: torch.Tensor, n_classes: int):
    """Unnormalized log posterior [N, K]:
    log prior[k] + sum_r mask[r,n] * log T_r[k, labels[r,n]]."""
    log_unnorm = torch.log(prior.clamp_min(_LOG_FLOOR)).unsqueeze(0).repeat(
        labels.shape[1], 1)
    for r in range(labels.shape[0]):
        # cell[n, k] = T_r[k, labels[r, n]]
        cell = confusions[r][:, labels[r].to(torch.int64)].T
        log_unnorm = log_unnorm + sel[r].to(torch.float64).unsqueeze(-1) * torch.log(
            cell.clamp_min(_LOG_FLOOR))
    return log_unnorm


def fit_dawid_skene(labels: torch.Tensor, mask: torch.Tensor, n_classes: int,
                    max_iter: int = 200, tol: float = 1e-8,
                    smoothing: float = 1e-8) -> dict:
    """Dawid-Skene EM. labels [R, N] long, mask [R, N] bool.

    Returns {"confusions": [R, K, K], "prior": [K], "posterior": [N, K],
    "log_likelihoods": [iterations], "iterations": int}. The returned posterior
    is the one under the returned parameters (the final E-step), so
    `rho_hat(posterior, ...)` reads a consistent pair. The likelihood is
    nondecreasing across `log_likelihoods` up to the smoothing perturbation of
    the M-step; the run stops at the first |delta| < tol.
    """
    sel = _validated(labels, mask, n_classes)
    if not 0 <= float(smoothing):
        raise ValueError("smoothing must be non-negative")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    posterior = majority_posterior(labels, mask, n_classes)
    prior, confusions = _m_step(posterior, labels, sel, n_classes, float(smoothing))
    history: list = []
    previous = None
    for _ in range(int(max_iter)):
        log_unnorm = _e_step(prior, confusions, labels, sel, n_classes)
        ll = float(torch.logsumexp(log_unnorm, dim=1).sum())
        history.append(ll)
        posterior = torch.softmax(log_unnorm, dim=1)
        if previous is not None and abs(ll - previous) < float(tol):
            break
        previous = ll
        prior, confusions = _m_step(posterior, labels, sel, n_classes, float(smoothing))
    return {
        "confusions": confusions,
        "prior": prior,
        "posterior": posterior,
        "log_likelihoods": torch.tensor(history, dtype=torch.float64),
        "iterations": len(history),
    }


def rho_hat(posterior: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
            n_classes: int) -> torch.Tensor:
    """Eq. (3) estimated from the EM posterior: rho_hat[k] in [0, 1], or NaN.

    rho_hat[k] = mean over all annotations (r, n) with mask = 1 and
    labels[r, n] = k of (1 - posterior[n, k]) — the posterior probability that
    this annotation disagrees with the consensus, conditioned on the annotation
    carrying label k exactly as Eq. (3) conditions. A category no such
    annotation carries has no estimate: NaN, never 0 and never invented.
    """
    sel = _validated(labels, mask, n_classes)
    if posterior.shape != (labels.shape[1], n_classes):
        raise ValueError(
            f"posterior must have shape ({labels.shape[1]}, {n_classes}); "
            f"got {tuple(posterior.shape)}")
    out = torch.full((n_classes,), float("nan"), dtype=torch.float64)
    for k in range(n_classes):
        pick = sel & (labels == k)
        if bool(pick.any()):
            cols = pick.nonzero(as_tuple=False)  # [M, 2] of (r, n)
            out[k] = (1 - posterior[cols[:, 1], k].to(torch.float64)).mean()
    return out


def majority_disagreement(labels: torch.Tensor, mask: torch.Tensor,
                          n_classes: int) -> dict:
    """The model-free cross-check: disagreement with the per-pixel majority.

    Returns {"per_category": [K], "overall": float}: the rate at which an
    annotation (r, n) with mask = 1 labeled k disagrees with the majority label
    of pixel n (ties: lowest class index), per category and overall. A category
    no annotation carries is NaN. This measures disagreement with the MAJORITY,
    not with the consensus k*, and is therefore not rho_k.
    """
    sel = _validated(labels, mask, n_classes)
    posterior = majority_posterior(labels, mask, n_classes)
    majority = posterior.argmax(dim=1)
    per_category = torch.full((n_classes,), float("nan"), dtype=torch.float64)
    for k in range(n_classes):
        pick = sel & (labels == k)
        if bool(pick.any()):
            cols = pick.nonzero(as_tuple=False)
            disagree = labels[cols[:, 0], cols[:, 1]].to(torch.int64) != majority[cols[:, 1]]
            per_category[k] = disagree.to(torch.float64).mean()
    if bool(sel.any()):
        cols = sel.nonzero(as_tuple=False)
        disagree = labels[cols[:, 0], cols[:, 1]].to(torch.int64) != majority[cols[:, 1]]
        overall = float(disagree.to(torch.float64).mean())
    else:
        overall = float("nan")
    return {"per_category": per_category, "overall": overall}
