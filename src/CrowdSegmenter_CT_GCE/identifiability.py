"""The three candidate identifiability constraints of r05 § "The loss".

r05 § "The loss" derives Eq. (2) and states: "Unconstrained, the reliability
branch is optimal at Λ = 1 exactly where P > C̃_k and at Λ = 0 elsewhere: it
would merely re-learn the truncation gate and carry no annotator information.
An explicit identifiability constraint on φ is therefore part of the objective
and not an optional regularizer. Three candidates are proposed, and the
synthetic selection study below decides among them; in every candidate φ is
trained against the reliability objective with the segmentation parameters held
fixed, never by (Ec. 1)." And § "Open items" carries the selection as open:
"The selection of the identifiability constraint among (Ec. 5), (Ec. 6) and
(Ec. 7), pending the synthetic study." No candidate is the design until the
study reports; this module implements the three exactly as written and invents
no fourth, no mixture and no improvement.

The three, verbatim in structure:

- C1, Eq. (5), mean-reliability normalization — each (annotator, category)
  spends the budget the noise estimation assigns to it:

      L_C1(φ) = Σ_r Σ_k ( E_{w,h}{Λ_r^k[w,h]} − (1 − ρ_{r,k}) )²,            (5)

  where ρ_{r,k} is "the flip rate of annotator r out of category k, derived
  from the confusion matrices of the noise estimation below". The EM's
  confusion matrices are T̂_r[k, k'] = P(k_r = k' | k* = k), so the flip rate
  out of category k is the off-diagonal mass of row k: ρ_{r,k} = 1 − T̂_r[k,k]
  (`flip_rates_from_confusions`). E_{w,h} is the plain mean over the entry
  axes, exactly as Eq. (5) writes it — no mask factor appears in Eq. (5).

- C2, Eq. (6), coupled likelihood with a fidelity-plus-trace penalty — the
  likelihood of the observed annotations under the symmetric-noise reliability
  model ("with probability Λ_r^k[w,h] the annotator writes the consensus
  category and otherwise one of the other K−1") plus the fidelity-plus-trace
  penalty of Zhang (2023), used with attribution, with weight λ > 0:

      L_C2(φ) = −E_r E_{w,h}{ M_r[w,h] · log( Λ_r^{k_r}[w,h] · P[w,h,k_r]
                + (1 − Λ_r^{k_r}[w,h]) · (1 − P[w,h,k_r])/(K−1) ) }
                + λ · E_r E_{w,h}{ M_r[w,h] · Λ_r^{k_r}[w,h] }.               (6)

- C3, Eq. (7), the coupled likelihood alone — the same likelihood without the
  trace penalty:

      L_C3(φ) = −E_r E_{w,h}{ M_r[w,h] · log( Λ_r^{k_r}[w,h] · P[w,h,k_r]
                + (1 − Λ_r^{k_r}[w,h]) · (1 − P[w,h,k_r])/(K−1) ) }.          (7)

The expectations E_r E_{w,h} follow the convention Eq. (1) fixes (`module.py`):
the mask sits INSIDE the expectation, so the value is the mean of the masked
integrand over every (r, w, h) entry.

`fit_reliability_branch` is the training step every candidate shares: φ is
trained against the chosen candidate's objective with the segmentation held
fixed — P enters only as a constant through the objective's closure and
nothing ever backpropagates into it. The trainer accepts only the three
declared candidates and fails closed on anything else: inventing a fourth
constraint is the decision r05 § "Open items" keeps with the deliberation.

One module per mathematical object; `__provenance__` is read statically by
`implementation_cli.py verify` — the module is never imported to obtain it —
and it is the only thing that makes revision drift detectable.
"""

from __future__ import annotations

from typing import Callable

import torch

from .reliability import annotated_labels, gather_at_annotated, validate_reliability

__provenance__ = {
    "revision": "research-concept-category-truncated-generalized-cross-entropy-for-semantic-se-r10.md",
    "sections": ["The loss"],
    "equations": ["5", "6", "7"],
    "invariants": [
        "c1_penalty_matches_manual_sum",
        "c1_gradient_vanishes_at_budget",
        "c2_likelihood_matches_manual",
        "c2_is_c3_plus_lambda_penalty",
        "c2_unit_reliability_is_neg_log_p",
        "c2_zero_reliability_is_uniform_flip",
        "flip_rates_are_offdiagonal_mass",
        "branch_training_never_touches_segmentation",
        "branch_trainer_fails_closed_on_unknown_candidate",
        "branch_trainer_requires_candidate_parameters",
    ],
}

#: Numerical slack for the closed-domain validations (values mathematically on
#: a boundary may land an ulp outside it).
RANGE_TOL = 1e-9
#: Slack for the probability-simplex check on P's rows.
SIMPLEX_TOL = 1e-6

#: The candidates r05 proposes, and the only names `fit_reliability_branch`
#: accepts. A constraint outside this set is not implemented anywhere: r05
#: § "Open items" keeps the selection, and any invention, with the
#: deliberation.
CANDIDATES = ("C1", "C2", "C3")


def _validated_p(P: torch.Tensor) -> int:
    """P's domain shared by Eqs. (6) and (7): [..., K] in [0, 1], rows sum to 1.

    Raises ValueError naming the failed precondition; returns the class count.
    """
    if not isinstance(P, torch.Tensor) or P.ndim < 2:
        raise ValueError(f"P must be a tensor of shape [..., K]; got {P!r}")
    n_classes = P.shape[-1]
    if n_classes < 2:
        raise ValueError(
            "Eqs. (6) and (7) divide by K - 1: at least two classes are "
            f"required; got K = {n_classes}")
    if not bool(torch.isfinite(P).all()):
        raise ValueError("P must be finite")
    if bool((P < -RANGE_TOL).any()) or bool((P > 1 + RANGE_TOL).any()):
        raise ValueError("P must lie in [0, 1]")
    row_sums = P.sum(dim=-1)
    if not bool((row_sums - 1).abs().max() <= SIMPLEX_TOL):
        raise ValueError("P rows must sum to 1 over the class axis")
    return n_classes


def _annotated_probabilities(P: torch.Tensor, Y: torch.Tensor, M: torch.Tensor,
                             Lambda: torch.Tensor):
    """Everything Eqs. (6) and (7) gather at the annotated class.

    Returns (p_annot, lambda_annot, m) with shapes [R, ...]: the model's
    probability and the reliability, both at the ANNOTATED category k_r(w,h)
    exactly as Eq. (1) gathers them, and M as a float tensor. Raises
    ValueError on any shape or domain violation.
    """
    n_classes = _validated_p(P)
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
    P_exp = P.unsqueeze(0).expand(M.shape[0], *P.shape)
    p_annot = gather_at_annotated(P_exp, labels)
    lambda_annot = gather_at_annotated(Lambda, labels)
    return p_annot, lambda_annot, M.to(P.dtype)


def likelihood_integrand(P: torch.Tensor, Y: torch.Tensor, M: torch.Tensor,
                         Lambda: torch.Tensor) -> torch.Tensor:
    """The masked log-likelihood integrand of Eqs. (6) and (7); shape [R, ...].

    `integrand[r, s] = M[r, s] * log( Λ_r^{k_r}[s] · P[s, k_r]
    + (1 − Λ_r^{k_r}[s]) · (1 − P[s, k_r])/(K − 1) )` — the symmetric-noise
    reliability model r05 states in prose: with probability Λ the annotator
    writes the consensus category and otherwise one of the other K−1. The
    negative mean of this tensor is Eq. (7)'s objective (`c3_loss`).

    The mixture weight is strictly positive wherever P lies strictly inside
    (0, 1) and Λ in [0, 1]; a pixel where the argument is non-positive (a
    one-hot P at the annotated class with Λ = 0) has no finite likelihood and
    is refused rather than logged as −inf.
    """
    p_annot, lambda_annot, m = _annotated_probabilities(P, Y, M, Lambda)
    n_classes = P.shape[-1]
    mixture = lambda_annot * p_annot + (1 - lambda_annot) * (1 - p_annot) / (n_classes - 1)
    if not bool((mixture > 0).all()):
        raise ValueError(
            "the symmetric-noise likelihood is non-positive at some annotated "
            "entry (one-hot P at the annotated class with Λ = 0 has no finite "
            "likelihood); P and Λ admit no log there")
    return m * torch.log(mixture)


def c3_loss(P: torch.Tensor, Y: torch.Tensor, M: torch.Tensor,
            Lambda: torch.Tensor) -> torch.Tensor:
    """Eq. (7), candidate C3: the coupled likelihood alone, a scalar tensor.

    `-likelihood_integrand.mean()` — the masked integrand averaged over every
    (r, w, h) entry, the convention Eq. (1) fixes and `module.py` states.
    """
    return -likelihood_integrand(P, Y, M, Lambda).mean()


def c2_penalty_integrand(P: torch.Tensor, Y: torch.Tensor, M: torch.Tensor,
                         Lambda: torch.Tensor) -> torch.Tensor:
    """The masked fidelity-plus-trace integrand of Eq. (6)'s second term.

    `integrand[r, s] = M[r, s] · Λ_r^{k_r}[s]`, gathered at the annotated
    class — the trace of the reliability over the annotated annotations, the
    penalty term of Zhang (2023) r05 cites with attribution.
    """
    p_annot, lambda_annot, m = _annotated_probabilities(P, Y, M, Lambda)
    return m * lambda_annot


def c2_loss(P: torch.Tensor, Y: torch.Tensor, M: torch.Tensor,
            Lambda: torch.Tensor, lam: float) -> torch.Tensor:
    """Eq. (6), candidate C2: likelihood plus λ-weighted penalty, scalar.

    `c3_loss(P, Y, M, Lambda) + lam * c2_penalty_integrand(...).mean()` —
    Eq. (6) is Eq. (7) plus the λ term, and the split is kept visible so the
    two candidates differ by exactly the penalty.
    """
    lam = float(lam)
    if not lam > 0:
        raise ValueError(f"the penalty weight λ must be > 0 (r05 Eq. (6)); got {lam!r}")
    likelihood = -likelihood_integrand(P, Y, M, Lambda).mean()
    penalty = c2_penalty_integrand(P, Y, M, Lambda).mean()
    return likelihood + lam * penalty


def flip_rates_from_confusions(confusions: torch.Tensor) -> torch.Tensor:
    """ρ_{r,k} of Eq. (5): the flip rate of annotator r OUT of category k.

    r05: "ρ_{r,k} is the flip rate of annotator r out of category k, derived
    from the confusion matrices of the noise estimation below." The EM's
    confusion matrices are T̂_r[k, k'] = P(k_r = k' | k* = k) with every row
    summing to one, so the flip rate out of k is the off-diagonal mass of
    row k: ρ_{r,k} = 1 − T̂_r[k,k]. Shape [R, K], values in [0, 1].

    Note the conditioning honestly: the confusion matrix conditions on the
    TRUE category k*, while r05 Eq. (3)'s ρ_k conditions on the ANNOTATED
    category. Eq. (5) names the confusion matrices as the source and this
    module implements exactly that; reconciling the two conditionings is the
    deliberation's call, not this code's.
    """
    if not isinstance(confusions, torch.Tensor) or confusions.ndim != 3:
        raise ValueError(f"confusions must be [R, K, K]; got {confusions!r}")
    r_annotators, k, k2 = confusions.shape
    if k != k2:
        raise ValueError(f"confusions must be square per annotator; got {tuple(confusions.shape)}")
    if not bool(torch.isfinite(confusions).all()):
        raise ValueError("confusions must be finite")
    if bool((confusions < -RANGE_TOL).any()) or bool((confusions > 1 + RANGE_TOL).any()):
        raise ValueError("confusions must lie in [0, 1]")
    if not bool((confusions.sum(dim=-1) - 1).abs().max() <= 1e-6):
        raise ValueError("every confusion row must sum to 1")
    return (1 - confusions.diagonal(dim1=-2, dim2=-1)).clamp(0, 1)


def c1_penalty(Lambda: torch.Tensor, rho_rk: torch.Tensor) -> torch.Tensor:
    """Eq. (5), candidate C1: the mean-reliability normalization, scalar.

    `Σ_r Σ_k ( E_{w,h}{Λ_r^k[w,h]} − (1 − ρ_{r,k}) )²` — the squared deviation
    of each (annotator, category) reliability's mean from the budget the noise
    estimation assigns, 1 − ρ_{r,k}. The expectation E_{w,h} is the plain mean
    over every entry axis of Λ beyond the annotator axis, exactly as Eq. (5)
    writes it: no mask factor appears in Eq. (5). `rho_rk` comes from
    `flip_rates_from_confusions` over the EM's confusion matrices.
    """
    if not isinstance(Lambda, torch.Tensor) or Lambda.ndim < 2:
        raise ValueError(f"Lambda must have shape [R, ..., K]; got {Lambda!r}")
    if (not isinstance(rho_rk, torch.Tensor)
            or rho_rk.shape != (Lambda.shape[0], Lambda.shape[-1])):
        raise ValueError(
            f"rho_rk must have shape ({Lambda.shape[0]}, {Lambda.shape[-1]}) "
            f"(annotators x classes); got {rho_rk!r}")
    if not bool(torch.isfinite(rho_rk).all()):
        raise ValueError("rho_rk must be finite")
    if bool((rho_rk < -RANGE_TOL).any()) or bool((rho_rk > 1 + RANGE_TOL).any()):
        raise ValueError("rho_rk must lie in [0, 1]")
    if not bool(torch.isfinite(Lambda).all()):
        raise ValueError("Lambda must be finite")
    if bool((Lambda < -RANGE_TOL).any()) or bool((Lambda > 1 + RANGE_TOL).any()):
        raise ValueError("Lambda must lie in [0, 1]")
    cell_means = Lambda.mean(dim=tuple(range(1, Lambda.ndim - 1)))  # [R, K]
    budget = 1 - rho_rk.to(Lambda.dtype)
    return ((cell_means - budget) ** 2).sum()


def fit_reliability_branch(objective: Callable[[torch.Tensor], torch.Tensor],
                           branch: torch.nn.Module, steps: int = 1500,
                           lr: float = 0.05) -> dict:
    """Train φ against the reliability objective, the segmentation held fixed.

    r05 § "The loss": "in every candidate φ is trained against the reliability
    objective with the segmentation parameters held fixed, never by (Ec. 1)."
    `objective` is a callable taking the branch's Λ tensor and returning the
    scalar loss of ONE of the three candidates (`c1_penalty` over fixed ρ_{r,k},
    `c2_loss` or `c3_loss` over fixed P, Y, M); the segmentation enters only as
    a constant through that closure, and this trainer never constructs Eq. (1)
    and never updates a segmentation parameter — the objective is the only
    signal, and Λ (the branch's output) the only thing optimized. `branch` is
    any module mapping the study's inputs to Λ with shape [R, ..., K].

    Full-batch Adam at `lr` for `steps` steps. Fails closed (RuntimeError) the
    moment the loss is non-finite: a NaN is a defect of the configuration, not
    a value to train through. Returns {"lambda": the trained Λ (detached),
    "logits": the trained parameters' tensors (detached), "losses": the
    per-step trajectory — whatever exists only while training runs is recorded
    now or it is gone —, "finalObjective": the candidate objective's value at
    the trained field, "steps", "lr"}.
    """
    if steps < 1:
        raise ValueError(f"steps must be at least 1; got {steps!r}")
    if not lr > 0:
        raise ValueError(f"lr must be positive; got {lr!r}")
    params = [p for p in branch.parameters() if p.requires_grad]
    if not params:
        raise ValueError("the branch has no trainable parameters")
    optimizer = torch.optim.Adam(params, lr=float(lr))
    losses: list = []
    for _ in range(int(steps)):
        optimizer.zero_grad()
        loss = objective(branch())
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(
                "the reliability objective produced a non-finite loss; the "
                "configuration is defective and training fails closed")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        trained = branch()
        final_objective = float(objective(trained))
    return {
        "lambda": trained.detach().clone(),
        "logits": [p.detach().clone() for p in params],
        "losses": torch.tensor(losses, dtype=torch.float64),
        "finalObjective": final_objective,
        "steps": int(steps),
        "lr": float(lr),
    }
