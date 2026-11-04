"""The CT-GCE loss: Eq. (1) and Eq. (2) of the bound revision.

r03 § "The loss" defines the Category-Truncated Generalized Cross-Entropy

    L(theta, phi) = E_r E_{w,h} { M_r[w,h] * (
        Lambda_r^{k_r}[w,h] * (1 - P[w,h,k_r]^q)/q
        + (1 - Lambda_r^{k_r}[w,h]) * (1 - C_{k_r}^q)/q ) }                 (1)

with per-category truncation thresholds C = (C_1, ..., C_K) and reliability
Lambda_r^k[w,h] in [0, 1] indexed by the annotated category, and its derivative

    d ell(w,h,r) / d Lambda_r^{k_r}[w,h]
        = M_r[w,h] * (C_{k_r}^q - P[w,h,k_r]^q) / q.                        (2)

Shapes (one image or a batch; S is the pixel/batch axes):

- probabilities P: [..., K], values in [0, 1], summing to one over the last dim;
- targets Y: one-hot [R, ..., K]; the annotated class is labels = Y.argmax(-1);
- selection mask M: [R, ...], values in {0, 1};
- reliability Lambda: [R, ..., K], values in [0, 1], indexed by the annotated class;
- cutoffs C: [K], values in (0, 1];
- q: float in (0, 1], required explicitly — there is no default.

The mask sits INSIDE the expectation: an unannotated pixel contributes zero to
the integrand and still counts in the denominator, so the loss is the mean over
all (r, w, h) of the masked integrand — `integrand.mean(dim=0).mean()`, i.e.
E_r then E_{w,h}.

One module per mathematical object; `__provenance__` is read statically by
`implementation_cli.py verify` — the module is never imported to obtain it —
and it is the only thing that makes revision drift detectable.
"""

from __future__ import annotations

import torch

from .reliability import annotated_labels, gather_at_annotated, validate_reliability

__provenance__ = {
    "revision": "research-concept-category-truncated-generalized-cross-entropy-for-semantic-se-r09.md",
    "sections": ["The loss"],
    "equations": ["1", "2"],
    "invariants": [
        "ctgce_nonnegative",
        "ctgce_unit_reliability_is_gce",
        "ctgce_zero_reliability_is_floor",
        "ctgce_is_affine_in_reliability",
        "ctgce_mask_zeroes_pixel",
        "ctgce_grad_lambda_matches_eq2",
        "ctgce_grad_p_matches_autograd",
    ],
}

#: Numerical slack for the closed-domain validations (values mathematically on
#: a boundary may land an ulp outside it).
RANGE_TOL = 1e-9
#: Slack for the probability-simplex check on P's rows.
SIMPLEX_TOL = 1e-6


def _validated(P: torch.Tensor, Y: torch.Tensor, M: torch.Tensor,
               Lambda: torch.Tensor, C: torch.Tensor, q: float):
    """Validate every precondition of Eq. (1) and return (labels, q, n_classes).

    Raises ValueError with a message naming the failed precondition. The
    proposal's domains, exactly: P in [0,1]^[...,K] with rows summing to one,
    Y one-hot [R,...,K], M in {0,1}^[R,...], Lambda in [0,1]^[R,...,K], C in
    (0,1]^[K], q in (0,1].
    """
    if not isinstance(P, torch.Tensor) or P.ndim < 2:
        raise ValueError(f"P must be a tensor of shape [..., K]; got {P!r}")
    n_classes = P.shape[-1]
    if n_classes < 1:
        raise ValueError("P must carry at least one class")
    if not bool(torch.isfinite(P).all()):
        raise ValueError("P must be finite")
    if bool((P < -RANGE_TOL).any()) or bool((P > 1 + RANGE_TOL).any()):
        raise ValueError("P must lie in [0, 1]")
    row_sums = P.sum(dim=-1)
    if not bool((row_sums - 1).abs().max() <= SIMPLEX_TOL):
        raise ValueError("P rows must sum to 1 over the class axis")

    labels = annotated_labels(Y)
    if Y.shape != labels.shape + (n_classes,):
        raise ValueError(
            f"Y must have shape {tuple(labels.shape) + (n_classes,)} to match P; "
            f"got {tuple(Y.shape)}")
    if M.shape != labels.shape:
        raise ValueError(f"M must have shape {tuple(labels.shape)}; got {tuple(M.shape)}")
    if not bool(((M == 0) | (M == 1)).all()):
        raise ValueError("M must be a selection mask with values exactly 0 or 1")
    validate_reliability(Lambda, labels, n_classes)
    if C.shape != (n_classes,):
        raise ValueError(f"C must have shape ({n_classes},); got {tuple(C.shape)}")
    if not bool(torch.isfinite(C).all()):
        raise ValueError("C must be finite")
    if bool((C <= 0).any()) or bool((C > 1).any()):
        raise ValueError("C must lie in (0, 1]")
    if isinstance(q, bool) or not isinstance(q, (int, float)):
        raise ValueError(f"q must be a float in (0, 1]; got {q!r}")
    q = float(q)
    if not 0 < q <= 1:
        raise ValueError(f"q must lie in (0, 1]; got {q!r}")
    return labels, q, n_classes


def ctgce_integrand(P: torch.Tensor, Y: torch.Tensor, M: torch.Tensor,
                    Lambda: torch.Tensor, C: torch.Tensor,
                    q: float) -> torch.Tensor:
    """The masked integrand of Eq. (1) at every (r, w, h); shape [R, ...].

    `integrand[r, s] = M[r, s] * ( lambda_annot * (1 - p_annot^q)/q
    + (1 - lambda_annot) * (1 - c_annot^q)/q )`, with p_annot, lambda_annot and
    c_annot gathered at the ANNOTATED class k_r(s). The loss is the mean of this
    tensor over r and then over the pixel/batch axes (`ctgce_loss`).
    """
    labels, q, _ = _validated(P, Y, M, Lambda, C, q)
    P_exp = P.unsqueeze(0).expand(M.shape[0], *P.shape)
    p_annot = gather_at_annotated(P_exp, labels)
    lambda_annot = gather_at_annotated(Lambda, labels)
    c_annot = C.to(P.dtype)[labels]
    m = M.to(P.dtype)
    return m * (lambda_annot * (1 - p_annot ** q) / q
                + (1 - lambda_annot) * (1 - c_annot ** q) / q)


def ctgce_loss(P: torch.Tensor, Y: torch.Tensor, M: torch.Tensor,
               Lambda: torch.Tensor, C: torch.Tensor, q: float) -> torch.Tensor:
    """Eq. (1): the CT-GCE loss, a scalar tensor.

    `integrand.mean(dim=0).mean()` — E_r first, then E_{w,h} — so an entry with
    M = 0 contributes zero and still counts in the denominator.
    """
    integrand = ctgce_integrand(P, Y, M, Lambda, C, q)
    return integrand.mean(dim=0).mean()


def ctgce_grad_lambda(P: torch.Tensor, Y: torch.Tensor, M: torch.Tensor,
                      C: torch.Tensor, q: float) -> torch.Tensor:
    """Eq. (2), the analytic derivative of the integrand w.r.t. Lambda.

    Shape [R, ..., K]: zero at every class except the annotated one, where it is
    `M * (c_annot^q - p_annot^q) / q`. Note the sign: positive where the model's
    probability at the annotated class sits below the cutoff (pushing Lambda
    down), negative where it sits above (pushing Lambda up) — the re-learning of
    the truncation gate r03 § "The loss" warns about.
    """
    if not isinstance(P, torch.Tensor) or P.ndim < 2:
        raise ValueError(f"P must be a tensor of shape [..., K]; got {P!r}")
    q_val = float(q)
    if not 0 < q_val <= 1:
        raise ValueError(f"q must lie in (0, 1]; got {q!r}")
    n_classes = P.shape[-1]
    labels = annotated_labels(Y)
    if Y.shape != labels.shape + (n_classes,):
        raise ValueError(
            f"Y must have shape {tuple(labels.shape) + (n_classes,)} to match P; "
            f"got {tuple(Y.shape)}")
    if M.shape != labels.shape:
        raise ValueError(f"M must have shape {tuple(labels.shape)}; got {tuple(M.shape)}")
    if C.shape != (n_classes,):
        raise ValueError(f"C must have shape ({n_classes},); got {tuple(C.shape)}")
    P_exp = P.unsqueeze(0).expand(M.shape[0], *P.shape)
    p_annot = gather_at_annotated(P_exp, labels)
    c_annot = C.to(P.dtype)[labels]
    g = M.to(P.dtype) * (c_annot ** q_val - p_annot ** q_val) / q_val
    out = torch.zeros(labels.shape + (n_classes,), dtype=P.dtype, device=P.device)
    out.scatter_(-1, labels.unsqueeze(-1), g.unsqueeze(-1))
    return out


def ctgce_grad_p(P: torch.Tensor, Y: torch.Tensor, M: torch.Tensor,
                 Lambda: torch.Tensor, C: torch.Tensor,
                 q: float) -> torch.Tensor:
    """The analytic derivative of the integrand's SUM w.r.t. P; shape [..., K].

    At the annotated class of entry (r, s) the derivative is
    `-M[r,s] * lambda_annot[r,s] * p_annot[r,s]^(q-1)`; zero at every other
    class. The annotator axis is scatter-added: P[s, k] receives the sum of the
    contributions of every annotator who annotated class k at s. Domain: the
    annotated probabilities must be strictly positive (q - 1 < 0 makes
    p^(q-1) singular at 0), which is the domain any softmax output lives in.

    Compared against autograd on the SUM of the integrand (never the mean) in
    `test_ctgce_grad_p_matches_autograd`.
    """
    labels, q, _ = _validated(P, Y, M, Lambda, C, q)
    P_exp = P.unsqueeze(0).expand(M.shape[0], *P.shape)
    p_annot = gather_at_annotated(P_exp, labels)
    lambda_annot = gather_at_annotated(Lambda, labels)
    if bool((p_annot <= 0).any()) and q < 1:
        raise ValueError("analytic grad w.r.t. P needs p_annot > 0 for q < 1")
    contrib = -M.to(P.dtype) * lambda_annot * p_annot ** (q - 1)
    out = torch.zeros(P.shape, dtype=P.dtype, device=P.device)
    flat_out = out.reshape(-1, P.shape[-1])
    flat_labels = labels.reshape(M.shape[0], -1)
    flat_contrib = contrib.reshape(M.shape[0], -1)
    for r in range(M.shape[0]):
        flat_out.scatter_add_(1, flat_labels[r].unsqueeze(-1), flat_contrib[r].unsqueeze(-1))
    return out
