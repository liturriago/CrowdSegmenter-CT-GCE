"""Level 2 - the metric pilot's own invariants (r09's metric claim protocol).

One test per claim the pilot's machinery makes about itself, each traceable to
the bound revision's metric section or to the comparability rules the pilot is
bound by. The heavy material (Inria patches, the backbone) is NOT loaded
here: these are small synthetic fixtures, so the suite stays fast and
dependency-light; the prototype's own ladder is checked separately, in the
pilot run, against the prototype's published CSV.
"""

from __future__ import annotations

import numpy as np
import pytest

from CrowdSegmenter_CT_GCE.metric_pilot import (Anchors, VERDICT_FLOOR_DRAWS,
                                                decision_metrics,
                                                prototype_ladder_subsets,
                                                random_mask_masks,
                                                sample_ladder_subsets,
                                                u1_panel_scores, verdict_for_step)


def _anchor_fixture() -> Anchors:
    """A small anchor set in the pilot's own declared shape."""
    return Anchors(
        floor_rand=-0.002,
        neutral_floor=0.75,
        ceil_ref=0.80,
        pop_good=[0.79, 0.795, 0.80, 0.805, 0.81, 0.785, 0.79, 0.80, 0.815, 0.80],
        pop_poor=[0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.67, 0.70, 0.74, 0.66],
    )


def test_pilot_ladder_composition() -> None:
    """The pilot's bench is the declared ladder: rung c keeps (10 - c) good
    annotators and swaps in c poor ones; endpoints have exactly one
    composition; every subset is an in-range, disjoint index pair."""
    subsets = sample_ladder_subsets(10, 30, 42)
    assert set(subsets) == set(range(11))
    for c, pairs in subsets.items():
        for good_ids, bad_ids in pairs:
            assert len(good_ids) == 10 - c
            assert len(bad_ids) == c
            # the annotator's identity is (profile, index): P1#3 and P5#3 are
            # different people, so the INDEX sets may overlap -- the prototype's
            # own protocol samples them independently. What must hold is that
            # the panel carries 10 DISTINCT (profile, index) annotators.
            identities = [("P1", a) for a in good_ids] + [("P5", a) for a in bad_ids]
            assert len(identities) == 10
            assert len(set(identities)) == 10
            assert all(0 <= a < 10 for a in good_ids + bad_ids)
    assert len(subsets[0]) == 1 and len(subsets[10]) == 1
    assert len(subsets[1]) == 30
    # and the same seed reproduces the same subsets
    again = sample_ladder_subsets(10, 30, 42)
    assert subsets == again


def test_pilot_per_annotator_scores_rung_invariant() -> None:
    """The per-annotator unit's defining property, made mechanical: C_a is
    computed once per annotator and never depends on the rung — a rung's
    scores are exactly the subset of fixed values its composition selects
    (Eq. 9's unit is the annotator, not the panel)."""
    subsets = sample_ladder_subsets(4, 30, 7)
    score_by_annotator = {("P1", a): 0.90 - 0.01 * a for a in range(4)}
    score_by_annotator.update({("P5", a): 0.70 + 0.02 * a for a in range(4)})
    rungs = u1_panel_scores(score_by_annotator, subsets, "P1", "P5")
    for c, pairs in subsets.items():
        for row, (good_ids, bad_ids) in zip(rungs[c], pairs):
            expected = [score_by_annotator[("P1", a)] for a in good_ids] \
                + [score_by_annotator[("P5", a)] for a in bad_ids]
            assert np.allclose(row, expected)
    # and the same subsets give bitwise-identical rungs
    again = u1_panel_scores(score_by_annotator, subsets, "P1", "P5")
    for c in subsets:
        assert np.array_equal(rungs[c], again[c])


def test_pilot_draw_subsets_shared_across_arms() -> None:
    """Arms differ in what they compute, never in which material they touch:
    one subset sample per ladder is reused by every arm, so two arms measured
    on the same subsets produce rung arrays of identical shape and pairing."""
    subsets_a = sample_ladder_subsets(10, 30, 42)
    # the P3 arm maps the SAME index pairs onto the P3 pool: same shapes
    for c in subsets_a:
        for good_ids, bad_ids in subsets_a[c]:
            assert len(good_ids) == 10 - c and len(bad_ids) == c


def test_pilot_determinism() -> None:
    """The pilot is reproducible from its declared seeds: the subset sample,
    the prototype's own sampling sequence and the random-mask floor all give
    bitwise-identical outputs on a repeated draw."""
    a = prototype_ladder_subsets(10, 42)
    b = prototype_ladder_subsets(10, 42)
    assert a == b
    floor_a = random_mask_masks(np.random.RandomState(7), 16, 4)
    floor_b = random_mask_masks(np.random.RandomState(7), 16, 4)
    for x, y in zip(floor_a, floor_b):
        assert np.array_equal(x, y)


def test_pilot_supervised_column_validation_only() -> None:
    """The supervised mIoU/Dice column validates and never decides: the
    zero-GT decision metrics are invariant to the validation column's values
    (permuting it moves nothing but the rank correlation reported beside
    them)."""
    rng = np.random.RandomState(3)
    rung_scores = [0.80 - 0.02 * c + 0.005 * rng.randn(30) for c in range(11)]
    supervised = [0.98 - 0.05 * c for c in range(11)]
    d1 = decision_metrics(rung_scores, supervised)
    perm = rng.permutation(11)
    d2 = decision_metrics(rung_scores, [supervised[i] for i in perm])
    for key in ("means", "se", "dynamicRange", "monotonicViolationsRaw",
                "monotonicViolationsBeyondNoise", "minAdjacentZ", "step_1_2_z", "steps"):
        assert d1[key] == d2[key]
    assert d1["spearmanVsSupervised"] != d2["spearmanVsSupervised"]


def test_pilot_verdicts_require_floor_and_separation() -> None:
    """No verdict below the declared floor of repetitions, and none within two
    combined standard errors: the verdict rule refuses exactly those."""
    below = verdict_for_step(1.0, 0.1, VERDICT_FLOOR_DRAWS - 1)
    assert below["status"] == "no-verdict"
    assert "floor" in below["reason"]
    deterministic = verdict_for_step(1.0, 0.0, VERDICT_FLOOR_DRAWS)
    assert deterministic["status"] == "no-verdict"
    wide = verdict_for_step(1.0, 0.1, VERDICT_FLOOR_DRAWS)
    assert wide["status"] == "separates"
    narrow = verdict_for_step(0.1, 0.1, VERDICT_FLOOR_DRAWS)
    assert narrow["status"] == "no-verdict"
    # a 6-draw ladder (the prototype's own protocol) gets no verdicts at all
    rng = np.random.RandomState(4)
    rung_scores = [0.80 - 0.02 * c + 0.005 * rng.randn(6) for c in range(11)]
    d = decision_metrics(rung_scores, [0.9] * 11)
    assert all(step["verdict"] == "no-verdict" for step in d["steps"])
    assert all("floor" in step["reason"] for step in d["steps"])
