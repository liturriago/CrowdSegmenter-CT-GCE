"""The metric pilot: the unit x normalization x summary study over the ladder.

r09 § "Falsifiable claim and assumptions" (Metric claim, zero-GT) leaves two
selections explicitly open, both to be closed by THIS pilot and by nobody else
until the operator's deliberation (r10):

- the normalization of `C_a`: two-point anchoring against the random-mask floor
  and the neighborhood-consistency reference (S1), standardization against a
  reference population (S2z / S2p), content-conditional normalization (S3) —
  all against the raw score (N0);
- the panel summary: the mean, the median, the lower quantile, and the
  fraction of annotators below a threshold.

and § "The evaluation metric" fixes the comparison's other half: the
per-annotator unit of Eq. (9) is the definition, and "the panel-consensus
variant measured by the first prototype is what the pilot compares it
against". The compression the prototype observed (0.7984 -> 0.7865 over the
contamination ladder) is the artifact the unit hypothesis blames on measuring
the consensus.

What this module implements, and what it refuses to do:

- the ladder is the prototype's own bench: 10 annotators, rung c replacing
  (10 - c) good (P1) with c poor (P5) annotators, c = 0..10, same data, same
  seeds. The P3 arm runs the identical ladder with the systematic-bias profile
  as the contaminant: it is the measurement of the revision's declared scope —
  consistency is not correctness — and it is reported, never hidden;
- every arm consumes the SAME draw subsets (one sample per ladder, reused by
  every unit, normalization and summary): arms differ in what they compute on
  the material, never in which material they touch;
- the decision metrics are zero-GT: dynamic range, monotonicity, adjacent-step
  separation, early detection. The supervised mIoU/Dice of the consensus
  against the dataset's ground truth is the VALIDATION column only — its rank
  correlation is reported, and `decision_metrics` is proven invariant to it
  (`pilot_supervised_column_validation_only`);
- nothing is selected here. Every result is per variant, provisional, and the
  deliberation closes the selection in r10.

Statistical units and floors, declared: 30 draws per rung (the declared
verdict floor of this repository's synthetic rounds); the composition subsets
are drawn once with seed 42. Rungs c = 0 and c = 10 have exactly one panel
composition, so their scores are deterministic and carry no dispersion — the
dispersion this study reads lives in the interior rungs. No verdict is granted
below the floor of repetitions or below a combined standard error of 2.

Aggregation axes, one at a time: a rung's mean collapses the draw axis
(panel composition); the ladder metrics collapse over rungs. Nothing pools
rungs and draws in one number, and nothing pools the two contaminant profiles.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

from .metric import (calibration_threshold, consistency, fraction_below,
                     panel_lower_quantile, panel_mean, panel_median, prototype)

__provenance__ = {
    "revision": "research-concept-category-truncated-generalized-cross-entropy-for-semantic-se-r09.md",
    "sections": [
        "The evaluation metric: neighborhood latent consistency",
        "Falsifiable claim and assumptions",
    ],
    "equations": ["8", "9"],
    "invariants": [
        "pilot_ladder_composition",
        "pilot_per_annotator_scores_rung_invariant",
        "pilot_draw_subsets_shared_across_arms",
        "pilot_determinism",
        "pilot_supervised_column_validation_only",
        "pilot_verdicts_require_floor_and_separation",
    ],
}

#: The verdict floor of repetitions: below this many draws per rung no verdict
#: is granted (the floor this repository's synthetic rounds declared).
VERDICT_FLOOR_DRAWS = 30

#: The z threshold past a combined standard error at which a step separates.
SEPARATION_Z = 2.0

VARIANTS = ("N0", "S1", "S2z", "S2p", "S3")
SUMMARIES = ("mean", "median", "q10", "fracBelow")


@dataclass(frozen=True)
class MetricPilotConfig:
    """The pilot's exact configuration; one instance is recorded in premises."""

    data_dir: str
    annotators_dir: str
    prototype_csv: str
    base_seed: int = 42
    subset_seed: int = 42
    n_annotators: int = 10
    good_profile: str = "P1"
    poor_profile: str = "P5"
    bias_profile: str = "P3"
    draws: int = 30
    quantile: float = 0.1
    threshold_k: float = 1.0
    n_random_masks: int = 32
    random_seed: int = 7
    disk_radius: int = 64


@dataclass
class Anchors:
    """The neighborhood's anchors, measured once, GT-free, panel-independent.

    - `floor_rand`: the mean consistency of random-mask "annotators" — what
      Eq. (9) scores a mask that carries no information, the prototype's own
      measured floor (its README: random masks sit at ~ -0.002 after
      centering);
    - `neutral_floor`: the consistency of prototypes under one FIXED neutral
      mask (a centered disk, identical on every patch) — the content term: the
      similarity any annotation inherits from the patches being visually
      similar. Seed-free and panel-free;
    - `ceil_ref`: the consistency of the neighborhood's reference annotation —
      the soft consensus of the full expert pool (all 10 P1), the best GT-free
      annotation available, fixed across the ladder. The reference's own
      movement with the measured panel is what the U2 ladder itself shows;
    - `pop_good`, `pop_poor`: the calibration populations in RAW score scale —
      the 10 expert and 10 poor annotators' C_a values. S2 standardizes
      against them; the threshold rule is derived from `pop_good` in every
      variant's own scale.
    """

    floor_rand: float
    neutral_floor: float
    ceil_ref: float
    pop_good: List[float]
    pop_poor: List[float]

    def normalized(self, c_values: np.ndarray, variant: str) -> np.ndarray:
        """Apply one normalization variant to raw scores; the four declared families."""
        c = np.asarray(c_values, dtype=np.float64)
        if variant == "N0":
            return c
        if variant == "S1":
            span = self.ceil_ref - self.floor_rand
            if span <= 0:
                raise ValueError("S1 needs ceil_ref > floor_rand; anchors are degenerate")
            return (c - self.floor_rand) / span
        if variant == "S3":
            span = self.ceil_ref - self.neutral_floor
            if span <= 0:
                raise ValueError("S3 needs ceil_ref > neutral_floor; anchors are degenerate")
            return (c - self.neutral_floor) / span
        if variant == "S2z":
            sigma = float(np.std(self.pop_good))
            if sigma <= 0:
                raise ValueError("S2z needs a spread-out expert population")
            return (c - float(np.mean(self.pop_good))) / sigma
        if variant == "S2p":
            pop = np.asarray(self.pop_good + self.pop_poor)
            flat = np.ravel(c)
            out = np.array([float(np.mean(pop <= v)) for v in flat])
            return out.reshape(np.shape(c))
        raise ValueError(f"unknown normalization variant {variant!r}")


def summarize(values: Sequence[float], summary: str, tau: float,
              quantile: float = 0.1) -> float:
    """One panel summary over one variant's normalized scores."""
    if summary == "mean":
        return panel_mean(values)
    if summary == "median":
        return panel_median(values)
    if summary == "q10":
        return panel_lower_quantile(values, quantile)
    if summary == "fracBelow":
        return fraction_below(values, tau)
    raise ValueError(f"unknown panel summary {summary!r}")


def panel_threshold(anchors: Anchors, variant: str, k: float) -> float:
    """The fraction-below threshold in one variant's own scale: mu_good - k*sigma_good."""
    pop = anchors.normalized(np.asarray(anchors.pop_good, dtype=np.float64), variant)
    return calibration_threshold(pop, k)


def panel_consensus_score(prototypes_by_image: Dict[str, np.ndarray],
                          image_ids: Sequence[str]) -> float:
    """The U2 unit: Eq. (9) over the panel's CONSENSUS masks — the prototype's variant.

    One prototype per patch (the consensus mask), one consistency over the
    neighborhood: exactly what the prototype measured, whose compressed ladder
    is the artifact under test. Ground truth enters nowhere.
    """
    protos = [prototypes_by_image[img] for img in image_ids]
    c, _ = consistency(protos)
    return c


def sample_ladder_subsets(n_annotators: int, draws: int, seed: int
                          ) -> Dict[int, List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]:
    """The ladder's composition subsets: one sample, shared by every arm.

    Rung c keeps (n - c) good annotators and swaps in c poor ones; the subsets
    are drawn from the combinations with one seeded `random.Random`, fixed
    before any arm runs and reused verbatim by every unit, normalization,
    summary and contaminant profile (the P3 arm maps the SAME index pairs onto
    the P3 pool). The two endpoints have exactly one composition.
    """
    rng = random.Random(seed)
    subsets: Dict[int, List[Tuple[Tuple[int, ...], Tuple[int, ...]]]] = {}
    for c in range(n_annotators + 1):
        combos_good = list(itertools.combinations(range(n_annotators), n_annotators - c))
        combos_bad = list(itertools.combinations(range(n_annotators), c))
        picked_good = rng.sample(combos_good, min(len(combos_good), draws))
        picked_bad = rng.sample(combos_bad, min(len(combos_bad), draws))
        n_draws = min(draws, len(combos_good) * len(combos_bad))
        subsets[c] = [
            (picked_good[d % len(picked_good)], picked_bad[d % len(picked_bad)])
            for d in range(n_draws)
        ]
    return subsets


def prototype_ladder_subsets(n_annotators: int, seed: int
                             ) -> Dict[int, List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]:
    """The prototype's own sampling sequence, reproduced verbatim for the port check.

    The prototype drew up to 6 subsets per rung with `random.Random(seed)` and
    paired them by `d % len`, sampling the good and poor combination lists in
    that order. Reproducing it exactly is what makes the port-fidelity pass a
    NUMERICS check (same compositions as the prototype's run): any difference
    against its published CSV is then device/precision drift, not sampling.
    """
    rng = random.Random(seed)
    subsets: Dict[int, List[Tuple[Tuple[int, ...], Tuple[int, ...]]]] = {}
    for c in range(n_annotators + 1):
        combos_good = list(itertools.combinations(range(n_annotators), n_annotators - c))
        combos_bad = list(itertools.combinations(range(n_annotators), c))
        s_good = rng.sample(combos_good, min(len(combos_good), 6))
        s_bad = rng.sample(combos_bad, min(len(combos_bad), 6))
        n_draws = min(6, len(combos_good) * len(combos_bad))
        subsets[c] = [
            (s_good[d % len(s_good)], s_bad[d % len(s_bad)])
            for d in range(n_draws)
        ]
    return subsets


def decision_metrics(rung_scores: List[np.ndarray],
                     supervised: Sequence[float]) -> Dict[str, object]:
    """The zero-GT decision metrics over one arm's ladder.

    `rung_scores[c]` is the array of per-draw panel scores at contamination
    c (the draw axis); `supervised[c]` is the rung's mean supervised quality —
    the validation column, which never enters these numbers. Collapses: the
    rung means collapse the draw axis (declared); the ladder metrics collapse
    over rungs. Direction is the caller's to orient: these numbers read
    higher-is-better.
    """
    if len(rung_scores) < 3:
        raise ValueError("the ladder metrics need at least three rungs")
    sizes = {arr.size for arr in rung_scores}
    if any(s < 1 for s in sizes):
        raise ValueError("every rung needs at least one draw")
    means = np.array([float(np.mean(a)) for a in rung_scores])
    ses = np.array([float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
                    for a in rung_scores])

    dynamic_range = float(means[0] - means[-1])

    violations_raw = int(np.sum(np.diff(means) > 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        step_se = np.sqrt(ses[:-1] ** 2 + ses[1:] ** 2)
    violation_mask = np.diff(means) > SEPARATION_Z * step_se
    violations_noise = int(np.sum(violation_mask & (step_se > 0)))

    rho, pval = spearmanr(means, np.asarray(supervised, dtype=np.float64))

    steps = [(float(means[c] - means[c + 1]),
              float(np.sqrt(ses[c] ** 2 + ses[c + 1] ** 2)))
             for c in range(len(means) - 1)]
    z_steps = [d / s if s > 0 else float("inf") if d > 0 else 0.0 for d, s in steps]
    min_z = float(min(z_steps))
    step_1_2_z = float(z_steps[1]) if len(z_steps) > 1 else float("nan")

    per_step = []
    for c, (d, s) in enumerate(steps):
        draws = min(rung_scores[c].size, rung_scores[c + 1].size)
        verdict = verdict_for_step(d, s, draws)
        per_step.append({
            "from_c": c, "to_c": c + 1,
            "delta": d, "se_combined": s,
            "z": z_steps[c],
            "verdict": verdict["status"],
            "reason": verdict["reason"],
        })

    return {
        "means": means.tolist(),
        "se": ses.tolist(),
        "dynamicRange": dynamic_range,
        "monotonicViolationsRaw": violations_raw,
        "monotonicViolationsBeyondNoise": violations_noise,
        "spearmanVsSupervised": None if np.isnan(rho) else float(rho),
        "spearmanPValue": None if pval is None or np.isnan(pval) else float(pval),
        "minAdjacentZ": min_z,
        "step_1_2_z": step_1_2_z,
        "steps": per_step,
    }


def u1_panel_scores(score_by_annotator: Dict[Tuple[str, int], float],
                    subsets: Dict[int, List[Tuple[Tuple[int, ...], Tuple[int, ...]]]],
                    good_profile: str, bad_profile: str) -> Dict[int, np.ndarray]:
    """Assemble the U1 unit's rung scores from the annotators' FIXED Eq. (9) values.

    The per-annotator unit's defining property, made mechanical here: `C_a` is
    computed once per annotator and never depends on the rung — a rung's panel
    scores are exactly the subset of the fixed values the composition selects.
    Returns {c: array of per-draw panel scores}; each draw's score is
    `summarize`'s business, applied by the caller over these arrays.
    """
    rungs: Dict[int, np.ndarray] = {}
    for c, pairs in subsets.items():
        rows = []
        for good_ids, bad_ids in pairs:
            rows.append([score_by_annotator[(good_profile, a)] for a in good_ids]
                        + [score_by_annotator[(bad_profile, a)] for a in bad_ids])
        rungs[c] = np.asarray(rows, dtype=np.float64)
    return rungs


def verdict_for_step(delta: float, se_combined: float, draws: int) -> Dict[str, str]:
    """The verdict rule: a step separates only past the floor AND past 2 combined SEs."""
    if draws < VERDICT_FLOOR_DRAWS:
        return {"status": "no-verdict",
                "reason": f"{draws} draws < the declared floor of {VERDICT_FLOOR_DRAWS}"}
    if se_combined == 0:
        return {"status": "no-verdict",
                "reason": "both rungs are deterministic single-composition endpoints"}
    if abs(delta) > SEPARATION_Z * se_combined:
        return {"status": "separates", "reason": ""}
    return {"status": "no-verdict",
            "reason": f"|delta| = {abs(delta):.3g} within {SEPARATION_Z} combined SEs "
                      f"({se_combined:.3g})"}


def early_detection(rung_scores: List[np.ndarray], upto: int = 3) -> Dict[str, Dict[str, float]]:
    """The early-detection read: how far the first contaminated rungs already move.

    For each summary, the z separation of rung 0 from rungs 1..`upto`,
    computed from THAT summary's own ladder. This is where the lower quantile
    and the mean are compared — the robustness question the pilot owes.
    """
    out: Dict[str, Dict[str, float]] = {}
    means = np.array([float(np.mean(a)) for a in rung_scores])
    ses = np.array([float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
                    for a in rung_scores])
    for c in range(1, min(upto, len(means) - 1) + 1):
        se = float(np.sqrt(ses[0] ** 2 + ses[c] ** 2))
        out[f"step_0_{c}"] = {"delta": float(means[0] - means[c]), "z": (means[0] - means[c]) / se if se > 0 else float("inf")}
    return out


def neutral_disk_mask(size: int, radius: int) -> np.ndarray:
    """The S3 neutral mask: a centered filled disk, identical on every patch."""
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = (size - 1) / 2.0
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2).astype(np.float32)


def random_mask_masks(rng: np.random.RandomState, size: int, count: int,
                      smooth_sigma: float = 8.0) -> List[np.ndarray]:
    """The S1 random-mask floor population: smooth random masks, no annotator signal.

    Smoothed (sigma = 8 at the 256 grid, a mid-scale blob the size of a
    building) and thresholded at the mean, so the masks are spatially
    structured the way a degenerate annotation is — not salt-and-pepper, whose
    prototypes the centering would flatten regardless. One mask per patch per
    synthetic annotator.
    """
    masks = []
    for _ in range(count):
        field = rng.randn(size, size)
        from scipy.ndimage import gaussian_filter
        field = gaussian_filter(field, sigma=smooth_sigma)
        masks.append((field > field.mean()).astype(np.float32))
    return masks
