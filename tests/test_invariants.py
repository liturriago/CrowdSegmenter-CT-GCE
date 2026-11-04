"""Level 2 - properties and invariants. The bridge between proposal and code.

One test per mathematical claim the proposal makes. The name MUST be
`test_<invariant_id>`, matching an id declared in a module's
`__provenance__["invariants"]`; verification pairs them by that exact name.

Each docstring cites the passage it enforces. A test whose claim cannot be
traced back to the proposal does not belong here.

Small deterministic tensors, CPU, float64: the claims are exact algebra, so the
comparisons run at TOL = 1e-10 and would not survive float32 rounding.
"""

from __future__ import annotations

import pytest
import torch

from CrowdSegmenter_CT_GCE.bound_study import (CrowdModel, bayes_risk,
                                               compare_cutoffs,
                                               consensus_posterior,
                                               expected_truncated_loss)
from CrowdSegmenter_CT_GCE.dawid_skene import (fit_dawid_skene,
                                               majority_disagreement)
from CrowdSegmenter_CT_GCE.module import (ctgce_grad_lambda, ctgce_grad_p,
                                          ctgce_integrand, ctgce_loss)
from CrowdSegmenter_CT_GCE.reliability import (
    IdentifiabilityConstraintRequired, annotated_labels,
    require_identifiability_constraint, unconstrained_optimum,
    validate_reliability)
from CrowdSegmenter_CT_GCE.thresholds import (calibrate_cutoffs,
                                              category_confidences,
                                              empirical_quantile)
from sweep import sweep

TOL = 1e-10


def _loss_fixture():
    """R = 2, N = 3, K = 3; one masked-out annotation (r=1, s=1)."""
    P = torch.tensor([[0.6, 0.3, 0.1],
                      [0.2, 0.5, 0.3],
                      [0.1, 0.1, 0.8]], dtype=torch.float64)
    Y = torch.zeros(2, 3, 3, dtype=torch.float64)
    for r, row in enumerate([[1, 0, 2], [2, 1, 0]]):
        for s, k in enumerate(row):
            Y[r, s, k] = 1.0
    M = torch.tensor([[1.0, 1.0, 1.0],
                      [1.0, 0.0, 1.0]], dtype=torch.float64)
    Lambda = torch.tensor([[[0.9, 0.5, 0.1], [0.2, 0.4, 0.6], [0.5, 0.5, 0.7]],
                           [[0.3, 0.8, 0.2], [0.6, 0.1, 0.4], [0.7, 0.2, 0.9]]],
                          dtype=torch.float64)
    C = torch.tensor([0.25, 0.40, 0.20], dtype=torch.float64)
    q = 0.7
    return P, Y, M, Lambda, C, q


def _threshold_fixture():
    """K = 2 (floor 1/2), one annotator, pixels 0..3 labeled 0, 4..7 labeled 1."""
    values0 = [0.10, 0.20, 0.30, 0.40]
    values1 = [0.20, 0.60, 0.90, 1.00]
    P = torch.zeros(8, 2, dtype=torch.float64)
    Y = torch.zeros(1, 8, 2, dtype=torch.float64)
    M = torch.ones(1, 8, dtype=torch.float64)
    for i, v in enumerate(values0):
        P[i, 0] = v
        P[i, 1] = 1 - v
        Y[0, i, 0] = 1.0
    for i, v in enumerate(values1):
        P[4 + i, 1] = v
        P[4 + i, 0] = 1 - v
        Y[0, 4 + i, 1] = 1.0
    return P, Y, M


def _ds_fixture():
    """R = 3, N = 4, K = 3, one masked-out annotation (r=1, n=3)."""
    labels = torch.tensor([[0, 1, 2, 1],
                           [0, 1, 0, 2],
                           [1, 1, 2, 0]], dtype=torch.int64)
    mask = torch.tensor([[True, True, True, True],
                         [True, True, True, False],
                         [True, True, True, True]])
    return labels, mask, 3


def _model_fixture(kind: str) -> CrowdModel:
    prior = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    if kind == "perfect":
        eye = torch.eye(3, dtype=torch.float64)
        confusions = torch.stack([eye, eye])
    elif kind == "uninformative":
        confusions = prior.unsqueeze(0).unsqueeze(0).repeat(2, 3, 1)
    else:
        confusions = torch.tensor(
            [[[0.6, 0.2, 0.2], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6]],
             [[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.1, 0.3, 0.6]]],
            dtype=torch.float64)
    return CrowdModel(prior=prior, confusions=confusions)


# ---------------------------------------------------------------------------
# module.py — Eq. (1) and Eq. (2)
# ---------------------------------------------------------------------------


def test_ctgce_nonnegative() -> None:
    """r03 § The loss, Eq. (1): both branches of the integrand are of the form
    (1 - x^q)/q with x in [0, 1] and q in (0, 1], hence nonnegative; the mask is
    0/1, so the loss — a mean of nonnegative terms — is nonnegative."""
    P, Y, M, Lambda, C, q = _loss_fixture()
    for lam in (Lambda, torch.zeros_like(Lambda), torch.ones_like(Lambda)):
        assert float(ctgce_loss(P, Y, M, lam, C, q)) >= 0.0
    zero_mask = torch.zeros_like(M)
    assert float(ctgce_loss(P, Y, zero_mask, Lambda, C, q)) == 0.0


def test_ctgce_unit_reliability_is_gce() -> None:
    """r03 § The loss, Eq. (1) at Lambda = 1: the floor branch vanishes and the
    loss is the masked mean of the generalized cross-entropy (1 - p^q)/q at the
    ANNOTATED class — computed here by explicit loops over (r, w, h)."""
    P, Y, M, Lambda, C, q = _loss_fixture()
    ones = torch.ones_like(Lambda)
    total, count = 0.0, 0
    for r in range(M.shape[0]):
        for s in range(M.shape[1]):
            k = int(Y[r, s].argmax())
            total += float(M[r, s]) * (1 - float(P[s, k]) ** q) / q
            count += 1
    expected = total / count
    got = float(ctgce_loss(P, Y, M, ones, C, q))
    assert abs(got - expected) < TOL


def test_ctgce_zero_reliability_is_floor() -> None:
    """r03 § The loss, Eq. (1) at Lambda = 0: the GCE branch vanishes and the
    loss is the masked mean of the category-wise floor (1 - C_k^q)/q, with k
    the ANNOTATED class — computed here by explicit loops over (r, w, h)."""
    P, Y, M, Lambda, C, q = _loss_fixture()
    zeros = torch.zeros_like(Lambda)
    total, count = 0.0, 0
    for r in range(M.shape[0]):
        for s in range(M.shape[1]):
            k = int(Y[r, s].argmax())
            total += float(M[r, s]) * (1 - float(C[k]) ** q) / q
            count += 1
    expected = total / count
    got = float(ctgce_loss(P, Y, M, zeros, C, q))
    assert abs(got - expected) < TOL


def test_ctgce_is_affine_in_reliability() -> None:
    """r03 § The loss ("(Ec. 1) is affine in the reliability"): for fixed
    P, Y, M, C, q the loss is affine in Lambda, so it commutes with every
    convex combination of two reliability fields."""
    P, Y, M, Lambda, C, q = _loss_fixture()
    other = torch.flip(Lambda, dims=(0, 2)).contiguous()
    alpha = 0.37
    left = float(ctgce_loss(P, Y, M, alpha * Lambda + (1 - alpha) * other, C, q))
    right = (alpha * float(ctgce_loss(P, Y, M, Lambda, C, q))
             + (1 - alpha) * float(ctgce_loss(P, Y, M, other, C, q)))
    assert abs(left - right) < TOL


def test_ctgce_mask_zeroes_pixel() -> None:
    """r03 § The loss, Eq. (1): M_r[w,h] sits inside the expectation, so zeroing
    one entry removes exactly that entry's integrand from the sum while keeping
    it in the denominator — the loss drops by integrand[r,s] / (R * N)."""
    P, Y, M, Lambda, C, q = _loss_fixture()
    r, s = 1, 2
    assert float(M[r, s]) == 1.0
    k = int(Y[r, s].argmax())
    p = float(P[s, k])
    lam = float(Lambda[r, s, k])
    c = float(C[k])
    contribution = lam * (1 - p ** q) / q + (1 - lam) * (1 - c ** q) / q
    assert contribution > 0.0
    M2 = M.clone()
    M2[r, s] = 0.0
    assert not bool(torch.equal(M2, M))
    full = float(ctgce_loss(P, Y, M, Lambda, C, q))
    zeroed = float(ctgce_loss(P, Y, M2, Lambda, C, q))
    assert abs((full - zeroed) - contribution / (M.shape[0] * M.shape[1])) < TOL


def test_ctgce_grad_lambda_matches_eq2() -> None:
    """r03 § The loss, Eq. (2): d(integrand)/dLambda at the annotated class is
    M * (C_k^q - p^q)/q and zero at every other class. Checked against the
    literal formula and against autograd on the SUM of the integrand."""
    P, Y, M, Lambda, C, q = _loss_fixture()
    got = ctgce_grad_lambda(P, Y, M, C, q)
    expected = torch.zeros_like(Lambda)
    for r in range(M.shape[0]):
        for s in range(M.shape[1]):
            k = int(Y[r, s].argmax())
            expected[r, s, k] = float(M[r, s]) * (float(C[k]) ** q - float(P[s, k]) ** q) / q
    assert bool((got - expected).abs().max() < TOL)

    lam = Lambda.clone().requires_grad_(True)
    out = ctgce_integrand(P, Y, M, lam, C, q)
    out.sum().backward()
    assert bool((lam.grad - expected).abs().max() < TOL)


def test_ctgce_grad_p_matches_autograd() -> None:
    """r03 § The loss, Eq. (1): the analytic derivative w.r.t. P at the
    annotated class is -M * lambda_annot * p^(q-1), scatter-added over
    annotators, zero at the other classes — matched against autograd on the SUM
    of the integrand (never the mean)."""
    P, Y, M, Lambda, C, q = _loss_fixture()
    p = P.clone().requires_grad_(True)
    out = ctgce_integrand(p, Y, M, Lambda, C, q)
    out.sum().backward()
    analytic = ctgce_grad_p(P, Y, M, Lambda, C, q)
    assert bool((p.grad - analytic).abs().max() < TOL)


# ---------------------------------------------------------------------------
# reliability.py — the reliability field and its open constraint
# ---------------------------------------------------------------------------


def test_reliability_field_range() -> None:
    """r03 § The loss: Lambda_r^k[w,h] in [0, 1]. The validation accepts the
    closed interval and refuses a value outside it (negative test: the mutated
    fixtures are asserted to actually differ from the valid one first)."""
    P, Y, M, Lambda, C, q = _loss_fixture()
    labels = annotated_labels(Y)
    assert validate_reliability(Lambda, labels, 3) is None
    for bad in (1.5, -0.1, float("nan")):
        mutated = Lambda.clone()
        mutated[0, 0, 0] = bad
        assert not bool(torch.equal(mutated, Lambda))
        with pytest.raises(ValueError):
            validate_reliability(mutated, labels, 3)


def test_reliability_indexed_by_annotated_category() -> None:
    """r03 § The loss ("indexed by the annotated category"): the loss reads the
    Lambda slot of the ANNOTATED class k_r(w,h). Raising that slot moves the
    loss; raising a non-annotated slot of the same entry does not."""
    P, Y, M, Lambda, C, q = _loss_fixture()
    base = torch.full_like(Lambda, 0.5)
    labels = annotated_labels(Y)
    k_ann = int(labels[0, 0])
    at_annotated = base.clone()
    at_annotated[0, 0, k_ann] = 0.9
    at_other = base.clone()
    at_other[0, 0, (k_ann + 1) % 3] = 0.9
    loss_base = float(ctgce_loss(P, Y, M, base, C, q))
    loss_ann = float(ctgce_loss(P, Y, M, at_annotated, C, q))
    loss_other = float(ctgce_loss(P, Y, M, at_other, C, q))
    assert abs(loss_other - loss_base) < TOL
    assert abs(loss_ann - loss_base) > TOL


def test_unconstrained_optimum_is_gate() -> None:
    """r03 § The loss, after Eq. (2): unconstrained, the reliability is optimal
    at Lambda = 1 exactly where P > C_k at the annotated class and at 0
    elsewhere — the gate, entry by entry."""
    P, Y, M, Lambda, C, q = _loss_fixture()
    gate = unconstrained_optimum(P, Y, M, C, q)
    expected = torch.zeros_like(gate)
    for r in range(M.shape[0]):
        for s in range(M.shape[1]):
            k = int(Y[r, s].argmax())
            if float(M[r, s]) == 1.0 and float(P[s, k]) > float(C[k]):
                expected[r, s, k] = 1.0
    assert torch.equal(gate, expected)


def test_unconstrained_optimum_is_hard_truncation() -> None:
    """r03 § The loss, Eq. (1) at the Eq. (2) optimum: the integrand collapses
    to min(GCE, floor) at every annotated entry — the hard truncation — so the
    loss equals the masked mean of that minimum (explicit loops)."""
    P, Y, M, Lambda, C, q = _loss_fixture()
    gate = unconstrained_optimum(P, Y, M, C, q)
    total, count = 0.0, 0
    for r in range(M.shape[0]):
        for s in range(M.shape[1]):
            k = int(Y[r, s].argmax())
            gce = (1 - float(P[s, k]) ** q) / q
            floor = (1 - float(C[k]) ** q) / q
            total += float(M[r, s]) * min(gce, floor)
            count += 1
    expected = total / count
    got = float(ctgce_loss(P, Y, M, gate, C, q))
    assert abs(got - expected) < TOL


def test_unconstrained_optimum_ignores_annotator_identity() -> None:
    """r03 § The loss: the unconstrained optimum of Eq. (2) depends on P, the
    annotated class and M only — never on which annotator produced the
    annotation. Two annotators labeling the same class at the same pixel get
    the same gate, and permuting the annotator order permutes the gate."""
    P, Y, M, Lambda, C, q = _loss_fixture()
    Y2 = Y.clone()
    Y2[1, 2, :] = 0.0
    Y2[1, 2, 2] = 1.0  # r1 now also annotates class 2 at pixel 2
    gate = unconstrained_optimum(P, Y2, M, C, q)
    assert float(gate[0, 2, 2]) == 1.0 and float(gate[1, 2, 2]) == 1.0
    assert torch.equal(gate[0, 2], gate[1, 2])

    perm = [1, 0]
    gate_perm = unconstrained_optimum(P, Y2[perm], M[perm], C, q)
    assert torch.equal(gate_perm, gate[perm])


def test_identifiability_guard_fails_closed() -> None:
    """r03 § The loss ("An explicit identifiability constraint on phi is
    therefore part of the objective and not an optional regularizer") and
    r03 § Open items ("The identifiability constraint on the reliability
    branch"): the constraint is an open item and may not be invented here, so
    the guard refuses `None` by name and passes anything else through."""
    with pytest.raises(IdentifiabilityConstraintRequired):
        require_identifiability_constraint(None)
    constraint = object()
    assert require_identifiability_constraint(constraint) is constraint


# ---------------------------------------------------------------------------
# thresholds.py — Eq. (4)
# ---------------------------------------------------------------------------


def test_cutoff_at_least_inverse_k() -> None:
    """r03 § Estimating the truncation thresholds, Eq. (4): the clipping at 1/K
    keeps C_k = max{1/K, F_k^{-1}(rho_hat_k)} at least 1/K — here the floor
    binds for category 0 and lifts its cutoff to exactly 1/K."""
    P, Y, M = _threshold_fixture()
    rho_hat = torch.tensor([0.25, 0.75], dtype=torch.float64)
    calibration = calibrate_cutoffs(P, Y, M, rho_hat)
    floor = 1.0 / 2
    assert bool((calibration.cutoffs >= floor - TOL).all())
    assert abs(float(calibration.cutoffs[0]) - floor) < TOL
    assert bool(calibration.clipped[0])


def test_cutoff_within_unit_interval() -> None:
    """r03 § Estimating the truncation thresholds, Eq. (4): the cutoffs live in
    [1/K, 1] — the quantile of values in [0, 1] cannot exceed 1 and the floor
    cannot exceed it either (rho_hat = 1 gives the maximum value)."""
    P, Y, M = _threshold_fixture()
    rho_hat = torch.tensor([1.0, 1.0], dtype=torch.float64)
    calibration = calibrate_cutoffs(P, Y, M, rho_hat)
    assert bool((calibration.cutoffs <= 1.0).all())
    assert bool((calibration.cutoffs >= 1.0 / 2 - TOL).all())


def test_cutoff_monotone_in_rho() -> None:
    """r03 § Estimating the truncation thresholds, Eq. (4): F_k^{-1} is
    nondecreasing in its level, so raising rho_hat_k cannot lower C_k — the
    floor only flattens, never reverses, the monotonicity."""
    P, Y, M = _threshold_fixture()
    low = calibrate_cutoffs(P, Y, M, torch.tensor([0.25, 0.25], dtype=torch.float64))
    high = calibrate_cutoffs(P, Y, M, torch.tensor([0.75, 0.75], dtype=torch.float64))
    assert bool((low.cutoffs <= high.cutoffs + TOL).all())
    assert not bool(torch.equal(low.cutoffs, high.cutoffs))


def test_quantile_matches_empirical_cdf() -> None:
    """r03 § Estimating the truncation thresholds, Eq. (4): the empirical
    generalized inverse (sort; index = ceil(level*n) - 1 clamped to [0, n-1];
    level = 0 is the minimum) satisfies the inverse-CDF inequalities
    F_n(q) >= level > F_n(q-) for every level in [0, 1]."""
    values = torch.tensor([0.05, 0.15, 0.25, 0.35, 0.45,
                           0.55, 0.65, 0.75, 0.85, 0.95], dtype=torch.float64)
    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        q_val = float(empirical_quantile(values, level))
        f_at = float((values <= q_val).to(torch.float64).mean())
        f_below = float((values < q_val).to(torch.float64).mean())
        assert f_at >= level - TOL
        if level > 0:
            assert f_below < level
    assert float(empirical_quantile(values, 0.0)) == float(values.min())
    assert float(empirical_quantile(values, 1.0)) == float(values.max())


def test_category_confidences_conditioned_on_annotated_label() -> None:
    """r03 § Estimating the truncation thresholds, Eq. (4): F_k conditions on
    k_r(w,h) = k. Category k's sample holds P[pixel, k] for every annotation
    (r, pixel) with M = 1 labeled k and nothing else — a pixel labeled j != k
    contributes nothing to category k even where P[pixel, k] is large, and a
    pixel with M = 0 is excluded outright."""
    P = torch.tensor([[0.77, 0.23],
                      [0.05, 0.95],
                      [0.60, 0.40]], dtype=torch.float64)
    Y = torch.zeros(2, 3, 2, dtype=torch.float64)
    Y[0, 0, 0] = 1.0   # annotated k = 0
    Y[0, 1, 0] = 1.0   # label present but M = 0: excluded
    Y[0, 2, 1] = 1.0   # annotated k = 1
    Y[1, 0, 1] = 1.0   # label present but M = 0: excluded
    Y[1, 1, 1] = 1.0   # annotated k = 1
    Y[1, 2, 0] = 1.0   # label present but M = 0: excluded
    M = torch.tensor([[1.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0]], dtype=torch.float64)
    samples = category_confidences(P, Y, M)
    expected = [[], []]
    for r in range(M.shape[0]):
        for s in range(M.shape[1]):
            if float(M[r, s]) == 1.0:
                k = int(Y[r, s].argmax())
                expected[k].append(float(P[s, k]))
    assert sorted(float(v) for v in samples[0]) == sorted(expected[0])
    assert sorted(float(v) for v in samples[1]) == sorted(expected[1])
    assert sorted(float(v) for v in samples[0]) == [0.77]
    assert sorted(float(v) for v in samples[1]) == [0.40, 0.95]


def test_calibration_requires_support() -> None:
    """r03 § Estimating the truncation thresholds, Eq. (4): a category with no
    annotated pixel has no F_k sample and a NaN rho_hat_k has no level — the
    calibration refuses both rather than inventing a threshold (negative tests,
    with a full-support positive control beside them)."""
    P, Y, M = _threshold_fixture()
    good = calibrate_cutoffs(P, Y, M, torch.tensor([0.25, 0.75], dtype=torch.float64))
    assert good.cutoffs.shape == (2,)

    Y2 = Y.clone()
    Y2[0, 4:8, :] = 0.0
    Y2[0, 4:8, 0] = 1.0  # category 1 loses every annotation
    assert category_confidences(P, Y2, M)[1].numel() == 0
    with pytest.raises(ValueError):
        calibrate_cutoffs(P, Y2, M, torch.tensor([0.25, 0.75], dtype=torch.float64))

    nan_rho = torch.tensor([0.25, float("nan")], dtype=torch.float64)
    with pytest.raises(ValueError):
        calibrate_cutoffs(P, Y, M, nan_rho)


def test_discarded_fraction_is_at_least_rho() -> None:
    """r04 § Estimating the truncation thresholds, Eq. (4): the honest discarded
    fraction is F_k(C_k), and it is at least rho_hat_k. The empirical quantile
    already satisfies F_k(F_k^{-1}(rho)) >= rho; the 1/K floor can only raise
    the cutoff, so it can only raise the discarded fraction. Equality holds
    when the floor does not bind and the quantile has no tie, as the r04 prose
    says ("up to ties at the quantile"). The direction is >=, never ==: the
    over-discard when the floor binds is by design."""
    # Equality case: K=2, floor 1/2; category 0 has four distinct confidences
    # and rho_hat=1/2, so the 1/2-quantile is 0.7, above the floor, and exactly
    # half the sample lies at or below it. Category 1 keeps the calibration
    # defined and over-discards, so the test also sees the strict side.
    P = torch.zeros(5, 2, dtype=torch.float64)
    Y = torch.zeros(1, 5, 2, dtype=torch.float64)
    M = torch.ones(1, 5, dtype=torch.float64)
    for i, value in enumerate([0.6, 0.7, 0.8, 0.9]):
        P[i, 0] = value
        P[i, 1] = 1.0 - value
        Y[0, i, 0] = 1.0
    P[4, 1] = 0.6
    P[4, 0] = 0.4
    Y[0, 4, 1] = 1.0
    rho_hat = torch.tensor([0.5, 0.5], dtype=torch.float64)
    calibration = calibrate_cutoffs(P, Y, M, rho_hat)
    assert not bool(calibration.clipped[0])
    assert abs(float(calibration.cutoffs[0]) - 0.7) < TOL
    assert abs(float(calibration.discarded[0]) - 0.5) < TOL
    assert float(calibration.discarded[1]) >= 0.5 - TOL

    # The declared sweep binds the floor in every category, so it exercises the
    # strict side: F_k(C_k) > rho_hat_k in 200/200 configurations.
    strict = 0
    for config in sweep():
        calibration = calibrate_cutoffs(config["P"], config["Y"], config["M"],
                                        config["rho_hat"])
        assert bool((calibration.discarded >= config["rho_hat"] - TOL).all())
        if bool((calibration.discarded > config["rho_hat"] + TOL).all()):
            strict += 1
    assert strict == 200


# ---------------------------------------------------------------------------
# dawid_skene.py — Eq. (3)
# ---------------------------------------------------------------------------


def test_posterior_normalized() -> None:
    """r03 § Estimating the truncation thresholds: the Dawid-Skene posterior
    over the consensus is a probability distribution per pixel — nonnegative
    rows summing to one after the log-space E-step's logsumexp normalization."""
    labels, mask, n_classes = _ds_fixture()
    fit = fit_dawid_skene(labels, mask, n_classes)
    posterior = fit["posterior"]
    assert bool((posterior >= 0).all())
    assert float((posterior.sum(dim=1) - 1).abs().max()) < TOL


def test_confusion_rows_normalized() -> None:
    """r03 § Estimating the truncation thresholds: T_r[k,k'] = P(k_r = k' |
    k* = k), so every row of every per-annotator confusion matrix sums to one
    after the smoothed, row-normalized M-step."""
    labels, mask, n_classes = _ds_fixture()
    fit = fit_dawid_skene(labels, mask, n_classes)
    confusions = fit["confusions"]
    assert confusions.shape == (labels.shape[0], n_classes, n_classes)
    assert float((confusions.sum(dim=-1) - 1).abs().max()) < TOL


def test_majority_disagreement_matches_manual_count() -> None:
    """r03 § Estimating the truncation thresholds: the model-free cross-check
    counts, per category and overall, the annotations disagreeing with the
    per-pixel majority (ties: lowest class index). Hand-counted on the fixture
    (the masked-out (r=1, n=3) is excluded): 3 disagreements among the 11
    masked annotations — (r=2, n=0), (r=1, n=2), (r=0, n=3) — and per label:
    label 0 -> 1/4 (only (r=1, n=2) disagrees), label 1 -> 2/5 (only (r=2, n=0)
    and (r=0, n=3) disagree), label 2 -> 0/2."""
    labels, mask, n_classes = _ds_fixture()
    got = majority_disagreement(labels, mask, n_classes)
    assert abs(got["overall"] - 3 / 11) < TOL
    per_category = [float(v) for v in got["per_category"]]
    assert abs(per_category[0] - 1 / 4) < TOL
    assert abs(per_category[1] - 2 / 5) < TOL
    assert abs(per_category[2] - 0.0) < TOL


# ---------------------------------------------------------------------------
# bound_study.py — the population pieces
# ---------------------------------------------------------------------------


def test_crowd_model_rows_normalized() -> None:
    """r03 § Falsifiable claim and assumptions: the crowd-noise model carries a
    prior summing to one and per-annotator confusions T_r[k,k'] = P(observed k'
    | truth k) with rows summing to one. A mutated row is refused (the mutation
    is asserted to have landed before the refusal is believed)."""
    model = _model_fixture("noisy")
    assert model.prior.shape == (3,)
    assert model.confusions.shape == (2, 3, 3)

    bad_prior = model.prior.clone()
    bad_prior[0] += 0.01
    assert not bool(torch.equal(bad_prior, model.prior))
    with pytest.raises(ValueError):
        CrowdModel(prior=bad_prior, confusions=model.confusions)

    bad_rows = model.confusions.clone()
    bad_rows[0, 0, 0] += 0.1
    assert not bool(torch.equal(bad_rows, model.confusions))
    with pytest.raises(ValueError):
        CrowdModel(prior=model.prior, confusions=bad_rows)


def test_bayes_risk_is_bounded() -> None:
    """r03 § Falsifiable claim and assumptions: the exact Bayes 0-1 risk given
    all R annotations is a probability of error, so it lies in [0, 1]. Anchored
    beyond the bounds: a perfect crowd has risk 0 and a fully uninformative
    crowd (rows equal to the prior) has risk 1 - max(prior)."""
    perfect = _model_fixture("perfect")
    risk_perfect = bayes_risk(perfect)
    assert 0.0 <= risk_perfect <= 1.0
    assert abs(risk_perfect - 0.0) < TOL

    uninformative = _model_fixture("uninformative")
    risk_uninformative = bayes_risk(uninformative)
    assert 0.0 <= risk_uninformative <= 1.0
    assert abs(risk_uninformative - (1 - 0.5)) < TOL

    noisy = _model_fixture("noisy")
    risk_noisy = bayes_risk(noisy)
    assert 0.0 <= risk_noisy <= 1.0


def test_global_cutoff_is_nested() -> None:
    """r03 § Falsifiable claim and assumptions: a single global cutoff is the
    per-category vector with every entry equal — the two expected truncated
    losses coincide exactly, and `compare_cutoffs`' difference is zero."""
    model = _model_fixture("noisy")
    predictor = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    q = 0.7
    c = 0.4
    per_category = torch.full((3,), c, dtype=torch.float64)
    result = compare_cutoffs(model, predictor, per_category, c, q)
    assert abs(float(result["difference"])) < TOL
    assert abs(float(result["per_category"]) - float(result["global"])) < TOL
    direct = expected_truncated_loss(model, predictor, per_category, q)
    assert abs(float(result["per_category"]) - float(direct)) < TOL


def test_consensus_posterior_without_annotations_is_prior() -> None:
    """r03 § Falsifiable claim and assumptions (A1): with no annotation at all
    the posterior over the consensus is the prior — the crowd likelihood
    contributes nothing when every mask entry is zero."""
    model = _model_fixture("noisy")
    labels = torch.zeros(2, 5, dtype=torch.int64)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    posterior = consensus_posterior(model, labels, mask)
    assert bool((posterior - model.prior.unsqueeze(0)).abs().max() < TOL)


# ---------------------------------------------------------------------------
# r05 § "The loss": the three candidate identifiability constraints.
# Every claim below cites the passage of r05 it enforces.
# ---------------------------------------------------------------------------

from CrowdSegmenter_CT_GCE.identifiability import (c1_penalty, c2_loss, c3_loss,
                                                   fit_reliability_branch,
                                                   flip_rates_from_confusions,
                                                   likelihood_integrand)
from CrowdSegmenter_CT_GCE.selection_study import (RegionTableBranch,
                                                   StudyConfig,
                                                   collapse_diagnosis,
                                                   estimate_thresholds,
                                                   generate_world,
                                                   recovery_error, run_cell,
                                                   train_candidate,
                                                   true_reliability_table)


def _candidates_fixture():
    """R = 2, N = 4, K = 3; one masked-out annotation (r=1, s=2)."""
    torch.manual_seed(11)
    gen = torch.Generator().manual_seed(11)
    P = torch.rand(4, 3, generator=gen, dtype=torch.float64) + 0.05
    P = P / P.sum(dim=-1, keepdim=True)
    Y = torch.zeros(2, 4, 3, dtype=torch.float64)
    for r, row in enumerate([[0, 1, 2, 1], [2, 0, 1, 0]]):
        for s, k in enumerate(row):
            Y[r, s, k] = 1.0
    M = torch.ones(2, 4, dtype=torch.float64)
    M[1, 2] = 0.0
    Lambda = torch.rand(2, 4, 3, generator=gen, dtype=torch.float64) * 0.8 + 0.1
    return P, Y, M, Lambda


def test_c1_penalty_matches_manual_sum() -> None:
    """r05 Eq. (5): L_C1 = sum_r sum_k ( E_{w,h}{Lambda_r^k} - (1 - rho_rk) )^2,
    the plain mean over the entry axes (Eq. (5) carries no mask factor),
    computed by hand on a small fixture."""
    torch.manual_seed(3)
    Lambda = torch.rand(2, 5, 3, generator=torch.Generator().manual_seed(3),
                        dtype=torch.float64)
    rho = torch.tensor([[0.2, 0.5, 0.7], [0.1, 0.3, 0.6]], dtype=torch.float64)
    manual = 0.0
    for r in range(2):
        for k in range(3):
            manual += (float(Lambda[r, :, k].mean()) - (1 - float(rho[r, k]))) ** 2
    assert abs(float(c1_penalty(Lambda, rho)) - manual) < TOL


def test_c1_gradient_vanishes_at_budget() -> None:
    """r05 Eq. (5): each (annotator, category) cell is exactly at its optimum
    when its mean equals the budget 1 - rho_rk — the gradient vanishes there
    and nowhere else, which is what 'spends the budget' means."""
    gen = torch.Generator().manual_seed(5)
    Lambda = (torch.rand(2, 6, 3, generator=gen, dtype=torch.float64) * 0.8 + 0.1
              ).requires_grad_(True)
    rho = torch.rand(2, 3, generator=gen, dtype=torch.float64) * 0.8 + 0.1
    # Put cell (1, 2) exactly at its budget.
    with torch.no_grad():
        Lambda[1, :, 2] = 1 - rho[1, 2]
    penalty = c1_penalty(Lambda, rho)
    grad, = torch.autograd.grad(penalty, Lambda)
    assert float(grad[1, :, 2].abs().max()) < TOL
    assert float(grad[0, :, :].abs().max()) > 1e-3
    assert float(grad[1, :, :2].abs().max()) > 1e-3


def test_c2_likelihood_matches_manual() -> None:
    """r05 Eq. (6), likelihood term: the masked integrand is
    M log( Lambda^{k_r} P[k_r] + (1 - Lambda^{k_r})(1 - P[k_r])/(K-1) ),
    gathered at the ANNOTATED class, computed by hand entry by entry."""
    P, Y, M, Lambda = _candidates_fixture()
    integrand = likelihood_integrand(P, Y, M, Lambda)
    labels = Y.argmax(-1)
    manual = torch.zeros(2, 4, dtype=torch.float64)
    for r in range(2):
        for s in range(4):
            k = int(labels[r, s])
            lam = float(Lambda[r, s, k])
            p = float(P[s, k])
            mixture = lam * p + (1 - lam) * (1 - p) / 2
            manual[r, s] = float(M[r, s]) * torch.log(
                torch.tensor(mixture, dtype=torch.float64))
    assert float((integrand - manual).abs().max()) < TOL


def test_c2_is_c3_plus_lambda_penalty() -> None:
    """r05 Eqs. (6) and (7): the candidates differ by exactly the penalty term
    — L_C2 = L_C3 + lambda * E{M Lambda^{k_r}} — for every weight, so the
    study's C2/C3 contrast is the trace penalty and nothing else."""
    P, Y, M, Lambda = _candidates_fixture()
    penalty = M * Lambda.gather(-1, Y.argmax(-1).unsqueeze(-1)).squeeze(-1)
    for lam in (0.05, 0.2, 0.8, 3.2):
        left = float(c2_loss(P, Y, M, Lambda, lam))
        right = float(c3_loss(P, Y, M, Lambda)) + lam * float(penalty.mean())
        assert abs(left - right) < TOL


def test_c2_unit_reliability_is_neg_log_p() -> None:
    """r05 § "The loss" prose: with probability Lambda the annotator writes the
    consensus category — at Lambda = 1 the likelihood of an annotation of
    category k_r is exactly the model's probability P[k_r], the noise-free
    limit of Eq. (6)'s mixture."""
    P, Y, M, _ = _candidates_fixture()
    ones = torch.ones_like(Y)
    labels = Y.argmax(-1)
    p_annot = P.unsqueeze(0).expand(2, *P.shape).gather(
        -1, labels.unsqueeze(-1)).squeeze(-1)
    expected = -(M * torch.log(p_annot)).mean()
    assert abs(float(c3_loss(P, Y, M, ones)) - float(expected)) < TOL


def test_c2_zero_reliability_is_uniform_flip() -> None:
    """r05 § "The loss" prose: otherwise one of the other K-1 — at Lambda = 0
    the likelihood of an annotation of category k_r is
    (1 - P[k_r])/(K-1), the uniform-flip limit of Eq. (6)'s mixture."""
    P, Y, M, _ = _candidates_fixture()
    zeros = torch.zeros_like(Y)
    labels = Y.argmax(-1)
    p_annot = P.unsqueeze(0).expand(2, *P.shape).gather(
        -1, labels.unsqueeze(-1)).squeeze(-1)
    expected = -(M * torch.log((1 - p_annot) / 2)).mean()
    assert abs(float(c3_loss(P, Y, M, zeros)) - float(expected)) < TOL


def test_flip_rates_are_offdiagonal_mass() -> None:
    """r05 Eq. (5) prose: rho_{r,k} is 'the flip rate of annotator r out of
    category k, derived from the confusion matrices' — the off-diagonal mass
    of row k, 1 - T_r[k,k], computed by hand."""
    confusions = torch.tensor([[[0.8, 0.15, 0.05], [0.1, 0.7, 0.2],
                                [0.05, 0.25, 0.7]],
                               [[0.9, 0.05, 0.05], [0.2, 0.6, 0.2],
                                [0.1, 0.1, 0.8]]], dtype=torch.float64)
    expected = 1 - confusions.diagonal(dim1=-2, dim2=-1)
    assert float((flip_rates_from_confusions(confusions) - expected).abs().max()) < TOL


def test_branch_training_never_touches_segmentation() -> None:
    """r05 § "The loss": 'in every candidate phi is trained against the
    reliability objective with the segmentation parameters held fixed, never
    by (Ec. 1)'. Training changes the branch, leaves P, Y and M bitwise
    identical, and the loss the trainer recorded is the candidate objective's
    value at the trained field — not a CT-GCE value."""
    config = StudyConfig(n_pixels=300, seeds=(0,))
    world = generate_world(config, "per-category", 0)
    thresholds = estimate_thresholds(world, config)
    P_snapshot = world["P"].clone()
    Y_snapshot = world["Y"].clone()
    M_snapshot = world["mask"].clone()
    result = train_candidate(world, thresholds, config, "C3")
    assert bool(torch.equal(world["P"], P_snapshot))
    assert bool(torch.equal(world["Y"], Y_snapshot))
    assert bool(torch.equal(world["mask"], M_snapshot))
    assert bool(torch.isfinite(result["losses"]).all())
    assert float((result["lambda"] - 0.5).abs().max()) > 1e-6  # training moved it
    recomputed = float(c3_loss(world["P"], world["Y"],
                               world["mask"].to(torch.float64),
                               result["lambda"]))
    assert abs(result["finalObjective"] - recomputed) < 1e-9


def test_branch_trainer_fails_closed_on_unknown_candidate() -> None:
    """r05 § "Open items": the selection among (Ec. 5), (Ec. 6) and (Ec. 7)
    is open, and 'no candidate is the design' — a trainer asked for a fourth
    constraint refuses instead of inventing one."""
    config = StudyConfig(n_pixels=300, seeds=(0,))
    world = generate_world(config, "per-category", 0)
    thresholds = estimate_thresholds(world, config)
    with pytest.raises(ValueError):
        train_candidate(world, thresholds, config, "C4")
    with pytest.raises(ValueError):
        train_candidate(world, thresholds, config, "c2")


def test_branch_trainer_requires_candidate_parameters() -> None:
    """r05 Eqs. (5) and (6): C1's budget rho_{r,k} and C2's weight lambda > 0
    are required inputs — a declared candidate's parameter is not defaulted,
    it is required (fail closed, no invented values)."""
    config = StudyConfig(n_pixels=300, seeds=(0,))
    world = generate_world(config, "per-category", 0)
    thresholds = estimate_thresholds(world, config)
    with pytest.raises(ValueError):
        train_candidate(world, thresholds, config, "C2", lam=None)
    with pytest.raises(ValueError):
        train_candidate(world, thresholds, config, "C2", lam=0.0)
    with pytest.raises(ValueError):
        c1_penalty(torch.rand(2, 5, 3, dtype=torch.float64),
                   torch.rand(2, 4, dtype=torch.float64))


# ---------------------------------------------------------------------------
# r05 § "The loss" derived study arms (round 2): S1/S2/S3, the consensus
# metrics, and the comparability contract with the round-1 world.
# ---------------------------------------------------------------------------

from CrowdSegmenter_CT_GCE.design_round import (DesignConfig, LinearSegmenter,
                                                arm_integrand, arm_pruning,
                                                consensus_metrics,
                                                generate_design_world,
                                                generate_features,
                                                surviving_lambda_mean)
import math


def _arms_fixture():
    """R = 2, N = 4, K = 3 — the candidates fixture with declared cutoffs."""
    P, Y, M, Lambda = _candidates_fixture()
    cutoffs = torch.tensor([0.35, 0.45, 0.30], dtype=torch.float64)
    return P, Y, M, Lambda, cutoffs


def test_s3_arm_is_eq1() -> None:
    """r05 Eq. (1): arm S3 is the proposal's loss exactly as written — its
    integrand equals `module.ctgce_integrand` entry for entry. The arm is the
    formulation, not a reimplementation of it."""
    P, Y, M, Lambda, cutoffs = _arms_fixture()
    q = 0.7
    eq1 = ctgce_integrand(P, Y, M, Lambda, cutoffs, q)
    s3 = arm_integrand("S3", P, Y, M, Lambda, cutoffs, q)
    assert float((eq1 - s3).abs().max()) < TOL


def test_s1_truncation_gradient_vanishes_on_pruned() -> None:
    """S1's mechanism is the thesis' truncation: where the gate is 0 the
    integrand is the constant floor, so P's gradient vanishes there; where the
    gate is 1 the full GCE gradient flows. Checked by autograd on a one-hot
    single-annotator fixture with one gated-out pixel."""
    logits = torch.tensor([[0.0, 4.0], [1.2, 0.0], [0.0, 4.0]],
                          dtype=torch.float64, requires_grad=True)
    P = torch.softmax(logits, dim=-1)
    Y = torch.zeros(1, 3, 2, dtype=torch.float64)
    for s, k in enumerate([1, 1, 1]):
        Y[0, s, k] = 1.0
    M = torch.ones(1, 3, dtype=torch.float64)
    Lambda = torch.full((1, 3, 2), 0.7, dtype=torch.float64)
    cutoffs = torch.tensor([0.5, 0.5], dtype=torch.float64)
    # Pixel 1's annotated-class probability is ~0.12 < 0.5: gated out.
    loss = arm_integrand("S1", P, Y, M, Lambda, cutoffs, 0.7).mean()
    grad, = torch.autograd.grad(loss, logits)
    assert float(grad[1].abs().max()) < TOL          # pruned: no gradient
    assert float(grad[0].abs().max()) > 1e-6         # survivor: GCE gradient
    assert float(grad[2].abs().max()) > 1e-6


def test_s2_survivor_weight_is_gate_times_lambda() -> None:
    """S2's delta is exactly the soft weight: its integrand equals S1's where
    Λ = 1, and differs from S1 by g·(Λ − 1)·GCE — the gate prunes (the hard
    decision) and Λ only re-weights the survivors."""
    P, Y, M, Lambda, cutoffs = _arms_fixture()
    q = 0.7
    s1 = arm_integrand("S1", P, Y, M, Lambda, cutoffs, q)
    s2 = arm_integrand("S2", P, Y, M, Lambda, cutoffs, q)
    labels = Y.argmax(-1)
    P_exp = P.unsqueeze(0).expand(2, *P.shape)
    p_annot = P_exp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    gate = (p_annot > cutoffs.to(P.dtype)[labels]).to(torch.float64)
    lam = Lambda.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    gce = (1 - p_annot ** q) / q
    floor = (1 - cutoffs.to(P.dtype)[labels] ** q) / q
    manual_delta = gate * (lam - 1) * (gce - floor)
    assert float((s2 - s1 - M * manual_delta).abs().max()) < TOL
    ones = torch.ones_like(Lambda)
    assert float((arm_integrand("S2", P, Y, M, ones, cutoffs, q)
                  - arm_integrand("S1", P, Y, M, ones, cutoffs, q)).abs().max()) < TOL


def test_consensus_metrics_match_manual() -> None:
    """The primary metric, counted by hand on a tiny fixture: per-class IoU and
    Dice of argmax P against the consensus labels, and the cross-entropy of P
    against the one-hot truth."""
    truth = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    P = torch.tensor([[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.25, 0.75]],
                     dtype=torch.float64)
    result = consensus_metrics(P, truth)
    # class 0: IoU 1/2, Dice 2/3; class 1: IoU 2/3, Dice 4/5.
    assert abs(result["mIoU"] - (0.5 + 2 / 3) / 2) < TOL
    assert abs(result["Dice"] - (2 / 3 + 0.8) / 2) < TOL
    ce = -(math.log(0.8) + math.log(0.4) + math.log(0.7) + math.log(0.75)) / 4
    assert abs(result["pErrorCE"] - ce) < TOL


def test_survivor_sets_match_manual() -> None:
    """Each arm's poda is its own mechanism, hand-counted: S1/S2 prune where
    the gate is 0, S3 where Λ < 0.5 (Eq. 1's handover); the surviving-Λ mean
    runs over each arm's own survivors, per category."""
    labels = torch.tensor([[1, 0, 1, 1]], dtype=torch.long)
    truth = torch.tensor([1, 0, 0, 0], dtype=torch.long)
    mask = torch.ones(1, 4, dtype=torch.bool)
    world = {
        "truth": truth, "region": torch.zeros(4, dtype=torch.long),
        "labels": labels, "mask": mask,
        "P": torch.tensor([[0.1, 0.9], [0.3, 0.7], [0.55, 0.45], [0.2, 0.8]],
                          dtype=torch.float64),
        "Y": torch.eye(2, dtype=torch.float64)[labels],
        "agree": labels == truth.unsqueeze(0),
        "lam_gen": torch.ones(1, 2, 1, dtype=torch.float64) * 0.75,
        "lam_true": torch.ones(1, 2, 1, dtype=torch.float64) * 0.75,
        "scenario": "manual", "seed": 0,
    }
    trained = torch.zeros(1, 4, 2, dtype=torch.float64)
    trained[0, 0, 1] = 0.9   # gate 1, Λ ≥ 0.5
    trained[0, 1, 0] = 0.2   # gate 0 (0.3 ≤ 0.5), Λ < 0.5
    trained[0, 2, 1] = 0.6   # gate 0 (0.45 ≤ 0.5), Λ ≥ 0.5
    trained[0, 3, 1] = 0.4   # gate 1 (0.8 > 0.5), Λ < 0.5
    cutoffs = torch.tensor([0.5, 0.5], dtype=torch.float64)
    # S1/S2 poda: pruned = {1, 2}; wrong = {2, 3} -> precision 1/2, recall 1/2.
    for arm in ("S1", "S2"):
        result = arm_pruning(world, trained, cutoffs, arm)
        assert result["prunedTotal"] == 2
        assert abs(result["precision"] - 0.5) < TOL
        assert abs(result["recall"] - 0.5) < TOL
    # S3 poda: pruned = {1, 3} -> precision 1/2, recall 1/2.
    s3 = arm_pruning(world, trained, cutoffs, "S3")
    assert s3["prunedTotal"] == 2
    assert abs(s3["precision"] - 0.5) < TOL
    assert abs(s3["recall"] - 0.5) < TOL
    # Surviving-Λ mean: S2's survivors are {0, 3} (both category 1) -> cat 1
    # mean (0.9 + 0.4)/2 = 0.65, cat 0 NaN. S3's survivors are {0, 2} -> cat 1
    # mean 0.75, cat 0 NaN.
    s2_mean = surviving_lambda_mean(world, trained, cutoffs, "S2")
    assert abs(s2_mean[1] - 0.65) < TOL
    assert s2_mean[0] != s2_mean[0]  # NaN, never invented
    s3_mean = surviving_lambda_mean(world, trained, cutoffs, "S3")
    assert abs(s3_mean[1] - 0.75) < TOL


def test_design_world_shares_round_one_world() -> None:
    """The comparability contract: the round-2 world IS the round-1 world for
    the same (scenario, seed) — every shared tensor bitwise identical — plus
    exactly one addition, the features."""
    config = StudyConfig(n_pixels=300, seeds=(0,))
    design = DesignConfig(base=config)
    a = generate_world(config, "per-category", 3)
    b = generate_design_world(design, "per-category", 3)
    for key in ("truth", "region", "labels", "mask", "P", "Y", "agree",
                "lam_gen", "lam_true"):
        assert bool(torch.equal(a[key], b[key])), key
    assert "features" in b and "features" not in a


def test_features_deterministic_per_seed() -> None:
    """The features are part of the declared world: the same seed draws the
    same features, a different seed draws different ones, and every feature
    row sits at its class's declared mean plus noise (finite)."""
    config = StudyConfig(n_pixels=300, seeds=(0,))
    design = DesignConfig(base=config)
    world = generate_world(config, "per-category", 3)
    a = generate_features(design, world["truth"], 3)
    b = generate_features(design, world["truth"], 3)
    c = generate_features(design, world["truth"], 4)
    assert bool(torch.equal(a, b))
    assert not bool(torch.equal(a, c))
    assert bool(torch.isfinite(a).all())


# ---------------------------------------------------------------------------
# The evaluation metric: Neighborhood Latent Consistency, r09 Eqs. (8) and (9)
# ---------------------------------------------------------------------------

import numpy as np

from CrowdSegmenter_CT_GCE.metric import (RING_RADIUS, boundary_ring,
                                          calibration_threshold, consistency,
                                          fraction_below, panel_lower_quantile,
                                          panel_mean, panel_median, prototype,
                                          prepare_features)


def _feature_fixture(seed: int = 0) -> np.ndarray:
    """A small prepared-ready feature map [C, H, W] with per-pixel structure."""
    rng = np.random.RandomState(seed)
    F = rng.randn(6, 16, 16).astype(np.float32)
    return F


def _mask_fixture() -> np.ndarray:
    """A small binary mask with a boundary, away from the image border."""
    Y = np.zeros((16, 16), dtype=np.float32)
    Y[4:12, 5:11] = 1.0
    return Y


def test_metric_prototype_unit_norm() -> None:
    """r09 Eq. (8): p(x, Y) is normalized — unit L2 norm overall and in each
    component, foreground and boundary alike, with the concatenation then
    normalized again."""
    F = prepare_features(_feature_fixture())
    Y = _mask_fixture()
    p = prototype(F, Y)
    assert p.shape == (12,)
    assert abs(float(np.linalg.norm(p)) - 1.0) < 1e-5
    fg_part, bnd_part = p[:6], p[6:]
    assert abs(float(np.linalg.norm(fg_part)) ** 2 + float(np.linalg.norm(bnd_part)) ** 2 - 1.0) < 1e-5
    # The manual recomposition of Eq. (8): norm([norm(p_fg), norm(p_bnd)]).
    B = boundary_ring(Y)
    eps = 1e-8
    p_fg = (F * Y).sum(axis=(1, 2)) / (Y.sum() + eps)
    p_bnd = (F * B).sum(axis=(1, 2)) / (B.sum() + eps)
    p_fg = p_fg / (np.linalg.norm(p_fg) + eps)
    p_bnd = p_bnd / (np.linalg.norm(p_bnd) + eps)
    p_manual = np.concatenate([p_fg, p_bnd])
    p_manual = p_manual / (np.linalg.norm(p_manual) + eps)
    assert np.abs(p - p_manual).max() < 1e-5


def test_metric_boundary_ring_dilate_minus_erode() -> None:
    """r09 Eq. (8): B_Y = dilate(Y) - erode(Y). Checked against a brute-force
    square-window morphological reference, including the ported zero-padding
    behaviour at the image border."""
    Y = np.zeros((9, 9), dtype=np.float32)
    Y[2:7, 3:6] = 1.0
    ring = boundary_ring(Y, radius=1)
    size = 2 * 1 + 1
    dilated = np.zeros_like(Y)
    eroded = np.zeros_like(Y)
    H, W = Y.shape
    for i in range(H):
        for j in range(W):
            i0, j0 = i - 1, j - 1
            crosses_border = i0 < 0 or j0 < 0 or i0 + size > H or j0 + size > W
            if crosses_border:
                # the ported zero padding: the window sees zeros, so the
                # erosion is 0 on any window crossing the border
                win = np.zeros((size, size), dtype=np.float32)
                for di in range(size):
                    for dj in range(size):
                        ii, jj = i0 + di, j0 + dj
                        if 0 <= ii < H and 0 <= jj < W:
                            win[di, dj] = Y[ii, jj]
                dilated[i, j] = win.max()
                eroded[i, j] = 0.0
            else:
                dilated[i, j] = Y[i0:i0 + size, j0:j0 + size].max()
                eroded[i, j] = Y[i0:i0 + size, j0:j0 + size].min()
    ring_manual = np.clip(dilated - eroded, 0.0, 1.0)
    assert np.abs(ring - ring_manual).max() < 1e-6
    assert ring.min() >= 0.0 and ring.max() <= 1.0


def test_metric_centering_kills_offset() -> None:
    """r09 Eq. (8)'s premise: F(x) spatially centered per pixel — a constant
    per-channel offset carries no information and must leave every prototype
    unchanged (the prototype's mean-leakage elimination)."""
    F = _feature_fixture()
    Y = _mask_fixture()
    offset = np.array([3.0, -2.0, 0.5, 10.0, -7.0, 1.0], dtype=np.float32).reshape(6, 1, 1)
    p_a = prototype(prepare_features(F), Y)
    p_b = prototype(prepare_features(F + offset), Y)
    assert np.abs(p_a - p_b).max() < 1e-6


def test_metric_consistency_matches_pairwise_average() -> None:
    """r09 Eq. (9): C_a is the mean pairwise cosine over i != j, divided by
    |N_a| (|N_a| - 1) — checked against the explicit double sum."""
    rng = np.random.RandomState(1)
    protos = [rng.randn(8) for _ in range(5)]
    protos = [p / np.linalg.norm(p) for p in protos]
    c, gram = consistency(protos)
    n = len(protos)
    total = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                total += float(protos[i] @ protos[j])
    assert abs(c - total / (n * (n - 1))) < 1e-10
    off_diag = gram.sum() - np.trace(gram)
    assert abs(c - off_diag / (n * (n - 1))) < 1e-10


def test_metric_consistency_bounds() -> None:
    """Eq. (9) is a mean of cosines of unit vectors, so C_a lies in [-1, 1]
    for any masks whatsoever."""
    rng = np.random.RandomState(2)
    for _ in range(20):
        protos = [rng.randn(8) for _ in range(4)]
        protos = [p / np.linalg.norm(p) for p in protos]
        c, _ = consistency(protos)
        assert -1.0 - 1e-12 <= c <= 1.0 + 1e-12


def test_metric_consistency_fails_closed_single_patch() -> None:
    """Eq. (9)'s divisor |N_a| (|N_a| - 1) is zero for |N_a| < 2: the equation
    is undefined there and the module fails closed. The prototype's 1.0
    fallback is deliberately not ported."""
    with pytest.raises(ValueError):
        consistency([np.eye(4)[0]])
    with pytest.raises(ValueError):
        consistency([])


def test_metric_panel_summaries_match_manual() -> None:
    """r09 § The evaluation metric: the four panel-summary candidates are the
    mean, the median, the lower quantile, and the fraction of annotators below
    a threshold; the threshold rule is mu - k*sigma of the calibration
    population."""
    values = [0.9, 0.85, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.2]
    assert abs(panel_mean(values) - float(np.mean(values))) < 1e-12
    assert abs(panel_median(values) - float(np.median(values))) < 1e-12
    assert abs(panel_lower_quantile(values, 0.1) - float(np.quantile(values, 0.1))) < 1e-12
    tau = calibration_threshold(values, 1.0)
    assert abs(tau - (float(np.mean(values)) - float(np.std(values)))) < 1e-12
    assert abs(fraction_below(values, tau) - float(np.mean(np.asarray(values) < tau))) < 1e-12
    with pytest.raises(ValueError):
        calibration_threshold([], 1.0)
    with pytest.raises(ValueError):
        panel_lower_quantile(values, 1.0)
