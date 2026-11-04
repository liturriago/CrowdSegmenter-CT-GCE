"""The reliability field, its annotated-class indexing, and its open constraint.

r03 § "The loss": ``Lambda_r^k[w,h] in [0,1]`` is "the reliability of annotator
r when annotating category k at pixel (w,h), indexed by the annotated category".
This module owns that indexing — which slot of the last axis the loss reads —
plus the range the field lives in and the pointwise unconstrained optimum that
r03 § "The loss", Eq. (2) derives.

r03 left the identifiability constraint on the reliability branch open; r05
proposes THREE candidates for it — C1, Eq. (5) (mean-reliability normalization
against the noise estimation's budget), C2, Eq. (6) (the coupled likelihood
plus Zhang (2023)'s fidelity-plus-trace penalty, weight λ), and C3, Eq. (7)
(the coupled likelihood alone) — and states: "in every candidate φ is trained
against the reliability objective with the segmentation parameters held fixed,
never by (Ec. 1)"; "Until the study reports, no candidate is the design." The
three candidates are implemented in `identifiability.py` and exercised by the
synthetic selection study (`selection_study.py`); which one the design adopts
is still the deliberation's decision, so `require_identifiability_constraint`
keeps failing closed on an unspecified constraint instead of picking one.

One module per mathematical object; `__provenance__` is read statically by
`implementation_cli.py verify` — the module is never imported to obtain it —
and it is the only thing that makes revision drift detectable.
"""

from __future__ import annotations

import torch

__provenance__ = {
    "revision": "research-concept-category-truncated-generalized-cross-entropy-for-semantic-se-r10.md",
    "sections": ["The loss", "Open items"],
    "equations": ["1", "2"],
    "invariants": [
        "reliability_field_range",
        "reliability_indexed_by_annotated_category",
        "unconstrained_optimum_is_gate",
        "unconstrained_optimum_is_hard_truncation",
        "unconstrained_optimum_ignores_annotator_identity",
        "identifiability_guard_fails_closed",
    ],
}

#: Numerical slack for the closed-interval validations below. The mathematical
#: domains are exact ([0, 1], (0, 1]); the slack only forgives float round-off
#: in a value that is mathematically on the boundary (a softmax row summing to
#: 1 - 1e-16). Anything a full ulp away from the boundary is still refused.
RANGE_TOL = 1e-9


class IdentifiabilityConstraintRequired(RuntimeError):
    """Raised by `require_identifiability_constraint` when the constraint the
    reliability branch is supposed to satisfy is unspecified (None). Named so
    callers can catch it; it is the fail-closed guard over r03's open item."""


def annotated_labels(Y: torch.Tensor) -> torch.Tensor:
    """The annotated category of every entry: ``labels[r, ...] = argmax_k Y[r, ..., k]``.

    r03 § "The loss": ``k_r(w,h)`` is "the category annotated by annotator r at
    (w,h)", and Y is one-hot, so the annotated category is the argmax over the
    last axis.

    Preconditions (ValueError): Y is a tensor with a class axis, its values are
    exactly in {0, 1}, and every row is one-hot (sums to one). The selection
    mask M, not Y, decides which entries the loss reads.
    """
    if not isinstance(Y, torch.Tensor):
        raise ValueError("Y must be a torch.Tensor of one-hot targets")
    if Y.ndim < 2:
        raise ValueError(f"Y must have shape [R, ..., K]; got {tuple(Y.shape)}")
    if not bool(((Y == 0) | (Y == 1)).all()):
        raise ValueError("Y must be one-hot: values exactly 0 or 1")
    if not bool((Y.sum(dim=-1) == 1).all()):
        raise ValueError("Y must be one-hot: every row sums to 1 over the class axis")
    return Y.argmax(dim=-1)


def gather_at_annotated(values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """The entry of `values` at the annotated class: ``[..., k] -> [...]``.

    `values` carries the class axis last ([R, ..., K] or [..., K] already
    expanded over annotators); `labels` is `annotated_labels`' output. This is
    the gather r03 § "The loss", Eq. (1) performs at the annotated category
    k_r(w,h) — never at the predicted category.
    """
    if values.shape[:-1] != labels.shape:
        raise ValueError(
            f"values leading shape {tuple(values.shape[:-1])} must equal "
            f"labels shape {tuple(labels.shape)}")
    return values.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def validate_reliability(Lambda: torch.Tensor, labels: torch.Tensor,
                         n_classes: int) -> None:
    """The reliability field's domain: shape [R, ..., K], values in [0, 1].

    r03 § "The loss": ``Lambda_r^k[w,h] in [0, 1]``, one entry per annotator,
    category and pixel, indexed by the annotated category. Raises ValueError on
    a wrong shape, a non-finite entry, or a value outside [0, 1].
    """
    if not isinstance(Lambda, torch.Tensor):
        raise ValueError("Lambda must be a torch.Tensor")
    expected = labels.shape + (n_classes,)
    if Lambda.shape != expected:
        raise ValueError(
            f"Lambda must have shape {expected} (annotators x entries x classes, "
            f"indexed by the annotated category); got {tuple(Lambda.shape)}")
    if not bool(torch.isfinite(Lambda).all()):
        raise ValueError("Lambda must be finite")
    if bool((Lambda < -RANGE_TOL).any()) or bool((Lambda > 1 + RANGE_TOL).any()):
        raise ValueError("Lambda must lie in [0, 1]")


def unconstrained_optimum(P: torch.Tensor, Y: torch.Tensor, M: torch.Tensor,
                          C: torch.Tensor, q: float) -> torch.Tensor:
    """The pointwise minimizer of Eq. (1) over Lambda, with P, Y, M, C fixed.

    r03 § "The loss", Eq. (2): the derivative w.r.t. Lambda at the annotated
    class is ``M_r[w,h] * (C_k^q - P[w,h,k]^q) / q``, which is negative exactly
    where P[w,h,k_r] > C_{k_r}. So the unconstrained optimum is the hard gate
    Lambda = 1 where the model's probability at the annotated class exceeds the
    category's cutoff and M = 1, and Lambda = 0 everywhere else (where the
    derivative is positive, or flat under M = 0). This function computes that
    gate; it is the *unconstrained* optimum r03 warns against ("it would merely
    re-learn the truncation gate and carry no annotator information"), not a
    reliability branch.
    """
    labels = annotated_labels(Y)
    n_classes = P.shape[-1]
    if P.shape[:-1] != labels.shape[1:]:
        raise ValueError(
            f"P must have shape [*S, {n_classes}] with labels [R, *S]; got "
            f"P {tuple(P.shape)}, labels {tuple(labels.shape)}")
    if C.shape != (n_classes,):
        raise ValueError(f"C must have shape ({n_classes},); got {tuple(C.shape)}")
    if not 0 < float(q) <= 1:
        raise ValueError(f"q must lie in (0, 1]; got {q!r}")
    if M.shape != labels.shape:
        raise ValueError(f"M must have shape {tuple(labels.shape)}; got {tuple(M.shape)}")
    p_annot = gather_at_annotated(P.unsqueeze(0).expand(M.shape[0], *P.shape), labels)
    c_annot = C.to(P.dtype)[labels]
    gate = (p_annot > c_annot) & (M != 0)
    out = torch.zeros(labels.shape + (n_classes,), dtype=P.dtype, device=P.device)
    out.scatter_(-1, labels.unsqueeze(-1), gate.to(P.dtype).unsqueeze(-1))
    return out


def require_identifiability_constraint(constraint) -> object:
    """Fail closed on the identifiability constraint the design has not adopted.

    r05 § "The loss" proposes three candidates for the constraint — (Ec. 5),
    (Ec. 6) and (Ec. 7) — and states: "Until the study reports, no candidate is
    the design." The synthetic selection study (r05 § "Falsifiable claim and
    assumptions") feeds the deliberation that closes the selection; until then
    the constraint is a REQUIRED input of any reliability training step and
    this guard refuses `None` rather than substituting one: adopting a
    candidate is the deliberation's decision, not this code's. A non-None
    constraint — one of r05's three candidates, declared by the caller — is
    returned unchanged.
    """
    if constraint is None:
        raise IdentifiabilityConstraintRequired(
            "the identifiability constraint on the reliability branch is a "
            "selection among r05's three candidates — C1 (Ec. 5), C2 (Ec. 6), "
            "C3 (Ec. 7) — that no candidate is the design until the synthetic "
            "study reports, and that the deliberation closes; pass the "
            "declared constraint instead of None")
    return constraint
