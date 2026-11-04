"""Level 3 - synthetic data. Deterministic, fixed seed, ground truth by construction.

State each expectation in the docstring BEFORE the assertion, so a passing test
cannot be mistaken for a hypothesis fitted after seeing the output.

Torch-only backend: every draw comes from the shared `rng` torch.Generator
fixture (conftest's fixed SEED) and every tensor is computed on CPU — the
Generator itself lives on CPU so a run on CUDA and a run on CPU draw the same
numbers and the tolerances below hold on both.
"""

from __future__ import annotations

import pytest
import torch

from CrowdSegmenter_CT_GCE.bound_study import (CrowdModel, bayes_risk,
                                               population_rho)
from CrowdSegmenter_CT_GCE.dawid_skene import (fit_dawid_skene, rho_hat)

SEED = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def rng() -> torch.Generator:
    """Suite-local deterministic generator: a fresh CPU torch.Generator seeded
    with the suite seed, identical draws to the origin conftest fixture."""
    return torch.Generator().manual_seed(SEED)
TOL_MONTE_CARLO = 0.01
TOL_RECOVERY = 0.05
MONTE_CARLO_SAMPLES = 200_000


def _crowd(rng, n_classes: int = 3, n_annotators: int = 4):
    """A valid crowd model with per-annotator per-category flip rates drawn in
    [0.1, 0.3] and flips uniform over the other categories. Returns
    (model, flips, prior)."""
    prior = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    flips = 0.1 + 0.2 * torch.rand(n_annotators, n_classes, generator=rng,
                                   dtype=torch.float64)
    confusions = torch.zeros(n_annotators, n_classes, n_classes, dtype=torch.float64)
    for r in range(n_annotators):
        for k in range(n_classes):
            confusions[r, k, k] = 1 - flips[r, k]
            for k2 in range(n_classes):
                if k2 != k:
                    confusions[r, k, k2] = flips[r, k] / (n_classes - 1)
    model = CrowdModel(prior=prior, confusions=confusions)
    return model, flips, prior


def _sample_annotations(rng, model, prior, n_pixels: int):
    """Ground truth by construction: y ~ prior, then every annotator keeps the
    truth with probability 1 - flip[r, y] or flips uniformly to another
    category. Returns (truth [N], labels [R, N])."""
    truth = torch.multinomial(prior, n_pixels, replacement=True, generator=rng)
    r_annotators, n_classes = model.confusions.shape[0], model.n_classes
    flip_prob = torch.stack([
        (1 - model.confusions[r].diagonal())[truth] for r in range(r_annotators)])
    is_flip = torch.rand(r_annotators, n_pixels, generator=rng,
                         dtype=torch.float64) < flip_prob
    alt = torch.randint(0, n_classes - 1, (r_annotators, n_pixels), generator=rng)
    alt = alt + (alt >= truth.unsqueeze(0)).to(torch.int64)
    labels = torch.where(is_flip, alt, truth.unsqueeze(0).expand(r_annotators, -1))
    return truth, labels.to(torch.int64)


def test_population_rho_matches_monte_carlo(rng) -> None:
    """Expectation: `population_rho` — rho_k = mean_r (1 - prior[k]*T_r[k,k] /
    sum_{k'} prior[k']*T_r[k',k]), the population form of Eq. (3) — matches the
    empirical P(k* != k | k_r = k) of a Monte Carlo sample of the same model,
    averaged over annotators, within TOL_MONTE_CARLO at 200k samples."""
    model, _, prior = _crowd(rng)
    exact = population_rho(model)

    truth = torch.multinomial(prior, MONTE_CARLO_SAMPLES, replacement=True,
                              generator=rng)
    estimated = torch.zeros(model.n_classes, dtype=torch.float64)
    for r in range(model.n_annotators):
        probs = model.confusions[r][truth]
        obs = torch.multinomial(probs, 1, replacement=True,
                                generator=rng).squeeze(1)
        for k in range(model.n_classes):
            sel = obs == k
            assert bool(sel.any()), "drawn sample must support every observed category"
            estimated[k] += float((truth[sel] != k).to(torch.float64).mean())
    estimated /= model.n_annotators

    assert float((estimated - exact).abs().max()) < TOL_MONTE_CARLO


def test_bayes_risk_matches_monte_carlo(rng) -> None:
    """Expectation: `bayes_risk` — the exact expected 0-1 risk of the Bayes
    predictor over all K^R observation vectors — matches the empirical error
    rate of that same predictor on a Monte Carlo sample of truth and
    annotations, within TOL_MONTE_CARLO at 200k samples."""
    model, _, prior = _crowd(rng)
    exact = bayes_risk(model)

    truth = torch.multinomial(prior, MONTE_CARLO_SAMPLES, replacement=True,
                              generator=rng)
    log_joint = torch.log(prior).unsqueeze(0).repeat(MONTE_CARLO_SAMPLES, 1)
    for r in range(model.n_annotators):
        probs = model.confusions[r][truth]
        obs = torch.multinomial(probs, 1, replacement=True,
                                generator=rng).squeeze(1)
        cells = model.confusions[r][:, obs].T  # [N, K]
        log_joint = log_joint + torch.log(cells)
    predicted = log_joint.argmax(dim=1)
    error_rate = float((predicted != truth).to(torch.float64).mean())

    assert abs(error_rate - exact) < TOL_MONTE_CARLO


def test_rho_recovers_synthetic_noise(rng) -> None:
    """Expectation: on a synthetic crowd whose per-annotator per-category flip
    rates are known by construction, the Dawid-Skene EM posterior plugged into
    Eq. (3)'s conditioning (rho_hat over annotations labeled k) recovers the
    population rho_k of the generating model within TOL_RECOVERY — the EM is
    initialized from the majority vote only to anchor the label permutation,
    not to bias the rates."""
    model, _, prior = _crowd(rng)
    n_classes = model.n_classes
    truth, labels = _sample_annotations(rng, model, prior, n_pixels=4000)
    mask = torch.ones_like(labels, dtype=torch.bool)

    # Ground truth by construction: Eq. (3) under the generating model,
    # written out here rather than taken from the code under test.
    rho_true = torch.zeros(n_classes, dtype=torch.float64)
    for k in range(n_classes):
        acc = 0.0
        for r in range(model.n_annotators):
            obs_mass = 0.0
            for k2 in range(n_classes):
                obs_mass += float(prior[k2]) * float(model.confusions[r, k2, k])
            keep_joint = float(prior[k]) * float(model.confusions[r, k, k])
            acc += 1 - keep_joint / obs_mass
        rho_true[k] = acc / model.n_annotators

    fit = fit_dawid_skene(labels, mask, n_classes)
    rho_est = rho_hat(fit["posterior"], labels, mask, n_classes)

    assert not bool(torch.isnan(rho_est).any())
    assert float((rho_est - rho_true).abs().max()) < TOL_RECOVERY


def test_em_log_likelihood_nondecreasing(rng) -> None:
    """Expectation: the EM log-likelihood history is nondecreasing — each
    M-step maximizes the expected complete log-likelihood, so the observed
    likelihood cannot fall. Slack of 1e-6 only: the smoothing term perturbs the
    exact maximizer, and the perturbation is bounded by it."""
    model, _, prior = _crowd(rng)
    _, labels = _sample_annotations(rng, model, prior, n_pixels=800)
    mask = torch.ones_like(labels, dtype=torch.bool)

    fit = fit_dawid_skene(labels, mask, model.n_classes)
    history = fit["log_likelihoods"]
    assert history.numel() >= 2
    assert bool((history[1:] - history[:-1] >= -1e-6).all())


# ---------------------------------------------------------------------------
# r05 § "Falsifiable claim and assumptions" ("Constraint selection (synthetic
# study)"): the synthetic world, its known truth, and the selection metrics.
# Expectations stated before the assertions; fixed seeds throughout.
# ---------------------------------------------------------------------------

from CrowdSegmenter_CT_GCE.reliability import unconstrained_optimum
from CrowdSegmenter_CT_GCE.selection_study import (StudyConfig, collapse_diagnosis,
                                                   estimate_thresholds,
                                                   floor_sensor,
                                                   gate_correlation,
                                                   generate_world,
                                                   pruning_precision_recall,
                                                   recovery_error,
                                                   true_reliability_table,
                                                   truth_gate_correlation)

SMALL = StudyConfig(n_pixels=300, seeds=(0,))


def test_truth_table_matches_manual_bayes() -> None:
    """Expectation: the true reliability table is the closed-form agreement
    probability given the ANNOTATED category — for each (r, k, g),
    pi_k * lam_gen[r,k,g] divided by the whole mass of annotations labeled k,
    computed here by explicit enumeration over the truths."""
    lam_gen = torch.tensor([[[0.9, 0.6], [0.5, 0.4], [0.8, 0.2], [0.3, 0.1]],
                            [[0.7, 0.7], [0.6, 0.5], [0.4, 0.9], [0.2, 0.3]]],
                           dtype=torch.float64)
    prior = (0.4, 0.3, 0.2, 0.1)
    k = lam_gen.shape[1]
    table = true_reliability_table(lam_gen, prior)
    for r in range(lam_gen.shape[0]):
        for cat in range(k):
            for g in range(lam_gen.shape[2]):
                agree = prior[cat] * float(lam_gen[r, cat, g])
                total = agree
                for c in range(k):
                    if c != cat:
                        total += prior[c] * (1 - float(lam_gen[r, c, g])) / (k - 1)
                assert abs(float(table[r, cat, g]) - agree / total) < TOL_MONTE_CARLO


def test_truth_table_in_unit_interval() -> None:
    """Expectation: the true reliability is a probability — every entry of the
    closed-form table lies in [0, 1] for all three declared scenarios."""
    config = StudyConfig()
    for scenario in ("region-varying", "per-category", "correlated-bias"):
        lam_gen = scenario_accuracy_table(config, scenario)
        table = true_reliability_table(lam_gen, config.prior)
        assert bool(torch.isfinite(table).all())
        assert float(table.min()) >= 0.0 and float(table.max()) <= 1.0


def scenario_accuracy_table(config, scenario):
    from CrowdSegmenter_CT_GCE.selection_study import scenario_accuracy
    return scenario_accuracy(config, scenario)


def test_generator_realizes_declared_agreement() -> None:
    """Ground truth by construction: the generator's realized fraction of
    agreeing annotations matches the declared per-(annotator, region)
    accuracy, and the realized agreement-given-annotated-category matches the
    closed-form truth table, within Monte-Carlo tolerance."""
    config = StudyConfig()  # the declared world, N = 4000
    world = generate_world(config, "region-varying", 7)
    agree = world["agree"].to(torch.float64)
    mask = world["mask"]
    for g in range(config.n_regions):
        in_region = mask & (world["region"].unsqueeze(0) == g)
        for r in range(config.n_annotators):
            declared = float(world["lam_gen"][r, :, g].mean())
            realized = float(agree[r][in_region[r]].mean())
            assert abs(realized - declared) < 0.05
    rows, cols = mask.nonzero(as_tuple=True)
    labels_at = world["labels"][rows, cols]
    region_at = world["region"][cols]
    true_at = world["lam_true"][rows, labels_at, region_at]
    agree_at = agree[rows, cols]
    for r in range(config.n_annotators):
        for k in range(config.n_classes):
            for g in range(config.n_regions):
                pick = (rows == r) & (labels_at == k) & (region_at == g)
                if int(pick.sum()) < 30:
                    continue
                assert abs(float(agree_at[pick].mean()) - float(true_at[pick].mean())) < 0.1


def test_recovery_error_zero_for_exact_recovery() -> None:
    """Expectation: the recovery error of the truth itself is exactly zero,
    and any perturbation of the field makes it positive — metric 1 measures
    distance to the truth and nothing else."""
    world = generate_world(SMALL, "per-category", 0)
    region = world["region"]
    exact = world["lam_true"][:, :, region].permute(0, 2, 1).contiguous()
    assert recovery_error(exact, world) == 0.0
    perturbed = exact * 0.9
    assert recovery_error(perturbed, world) > 0.0


def test_gate_correlation_of_the_gate_is_one() -> None:
    """r05 § "The loss": unconstrained, the reliability branch's optimum IS the
    truncation gate — so the gate field's correlation with the gate is 1, the
    reference metric 2 reads the candidates against. The truth's own
    correlation is strictly lower and finite."""
    config = SMALL
    world = generate_world(config, "per-category", 0)
    thresholds = estimate_thresholds(world, config)
    gate = unconstrained_optimum(world["P"], world["Y"],
                                 world["mask"].to(torch.float64),
                                 thresholds["cutoffs"], config.q_unconstrained)
    correlation = gate_correlation(gate, world, thresholds["cutoffs"])
    assert not correlation != correlation  # not NaN: the gate has variance on E
    assert abs(correlation - 1.0) < 1e-9
    truth_corr = truth_gate_correlation(config, "per-category", 0)
    assert truth_corr < 1.0


def test_collapse_detector_fires_on_saturated_field() -> None:
    """Expectation: the declared collapse predicate fires on a saturated field
    (every entry within 0.02 of 0 or 1) and on a dead flat field (zero
    standard deviation), and does not fire on a spread field — the three
    shapes the pilot must tell apart."""
    world = generate_world(SMALL, "per-category", 0)
    rows, cols = world["mask"].nonzero(as_tuple=True)
    labels_at = world["labels"][rows, cols]
    shape = world["mask"].shape + (world["P"].shape[-1],)
    saturated = torch.zeros(shape, dtype=torch.float64)
    saturated[rows, cols, labels_at] = (torch.arange(int(rows.numel())) % 2 == 0).to(
        torch.float64)
    assert collapse_diagnosis(saturated, world)["collapsed"]
    flat = torch.full(shape, 0.5, dtype=torch.float64)
    assert collapse_diagnosis(flat, world)["collapsed"]
    spread = torch.full(shape, 0.5, dtype=torch.float64)
    spread = spread + 0.2 * torch.sin(torch.arange(shape[1], dtype=torch.float64)).view(
        1, -1, 1)
    assert not collapse_diagnosis(spread, world)["collapsed"]


def test_pruning_matches_manual_count() -> None:
    """Expectation: the pruning prunes exactly the declared per-category budget
    ceil(rho_hat_k * count_k), and precision/recall equal the hand-counted
    ratios on a world whose wrong annotations are known by construction."""
    shape_labels = torch.tensor([[0, 0, 0, 1]], dtype=torch.long)
    truth = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    mask = torch.ones(1, 4, dtype=torch.bool)
    world = {
        "truth": truth,
        "region": torch.zeros(4, dtype=torch.long),
        "labels": shape_labels,
        "mask": mask,
        "P": torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.55, 0.45], [0.2, 0.8]],
                          dtype=torch.float64),
        "Y": torch.eye(2, dtype=torch.float64)[shape_labels],
        "agree": shape_labels == truth.unsqueeze(0),
        "lam_gen": torch.ones(1, 2, 1, dtype=torch.float64) * 0.75,
        "lam_true": torch.ones(1, 2, 1, dtype=torch.float64) * 0.75,
        "scenario": "manual", "seed": 0,
    }
    trained = torch.zeros(1, 4, 2, dtype=torch.float64)
    trained[0, 0, 0] = 0.1  # category-0 annotations ranked 0.1, 0.3, 0.5
    trained[0, 1, 0] = 0.3
    trained[0, 2, 0] = 0.5
    trained[0, 3, 1] = 0.9
    rho = torch.tensor([0.5, 0.0], dtype=torch.float64)
    result = pruning_precision_recall(trained, world, rho)
    # Category 0: 3 annotations, budget ceil(0.5 * 3) = 2, pruned the two
    # lowest (0.1 correct, 0.3 wrong) -> prunedWrong 1, precision 0.5,
    # recall 1/1 = 1. Category 1: budget 0 -> precision NaN.
    cat0, cat1 = result["perCategory"]
    assert cat0["budget"] == 2 and cat0["prunedWrong"] == 1
    assert abs(cat0["precision"] - 0.5) < TOL_MONTE_CARLO
    assert abs(cat0["recall"] - 1.0) < TOL_MONTE_CARLO
    assert cat1["budget"] == 0
    assert cat1["precision"] != cat1["precision"]  # NaN, never invented
    assert abs(result["precision"] - 0.5) < TOL_MONTE_CARLO
    assert abs(result["recall"] - 1.0) < TOL_MONTE_CARLO


def test_floor_sensor_matches_manual_cdf() -> None:
    """r05 Eq. (4)'s floor: the case-A sensor F_k(1/K) equals the fraction of
    category-k annotated confidences at or below 1/K, counted by hand."""
    world = generate_world(SMALL, "per-category", 0)
    thresholds = estimate_thresholds(world, config=SMALL)
    sensor = floor_sensor(world, thresholds)
    k = world["P"].shape[-1]
    floor = 1.0 / k
    rows, cols = world["mask"].nonzero(as_tuple=True)
    labels_at = world["labels"][rows, cols]
    p_annot = world["P"][cols, :]
    for cat in range(k):
        pick = labels_at == cat
        manual = float((p_annot[pick, cat] <= floor).to(torch.float64).mean())
        assert abs(sensor["fFloor"][cat] - manual) < 1e-12


def test_protocol_shares_data_across_candidates() -> None:
    """The study's contract: same protocol, same data — the world a seed draws
    is bitwise identical however many candidates train on it, and a different
    seed draws different material."""
    a = generate_world(SMALL, "per-category", 3)
    b = generate_world(SMALL, "per-category", 3)
    c = generate_world(SMALL, "per-category", 4)
    for key in ("truth", "region", "labels", "mask", "P", "Y", "agree"):
        assert bool(torch.equal(a[key], b[key])), key
    assert not bool(torch.equal(a["labels"], c["labels"]))


def test_pilot_cell_runs_end_to_end() -> None:
    """Expectation: one full run of the declared protocol — world, warm-up
    estimation, candidate training, the four metrics — returns a record whose
    metrics are finite where they must be, for each of the three candidates."""
    config = SMALL
    world = generate_world(config, "correlated-bias", 0)
    thresholds = estimate_thresholds(world, config)
    for candidate, lam in (("C1", None), ("C2", 0.8), ("C3", None)):
        from CrowdSegmenter_CT_GCE.selection_study import train_candidate as tc
        trained = tc(world, thresholds, config, candidate, lam)
        record_metrics = {
            "recoveryError": recovery_error(trained["lambda"], world),
            "gateCorrelation": gate_correlation(trained["lambda"], world,
                                                thresholds["cutoffs"]),
            "collapse": collapse_diagnosis(trained["lambda"], world),
            "pruning": pruning_precision_recall(trained["lambda"], world,
                                                thresholds["rho_hat"]),
        }
        assert record_metrics["recoveryError"] >= 0.0
        assert record_metrics["recoveryError"] < 1.0
        assert record_metrics["pruning"]["prunedTotal"] > 0
