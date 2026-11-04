"""The metric pilot's run orchestration: material in, per-variant results out.

This module implements no equation — the equations are
`CrowdSegmenter_CT_GCE.metric` (Eqs. 8-9) and the arms are
`CrowdSegmenter_CT_GCE.metric_pilot`. Everything here is the harness's
material and wiring, ported from the first prototype (`metric_material.py`)
and sequenced once so the notebook stays thin and the run is reproducible from
one entry point:

1. load the prototype's 15 Inria patches and their ground truth;
2. extract each patch's prepared dense features (the prototype's ResNet-34
   chain, CPU; the chain is baked into the extractor, so `prepare_features`
   is NOT applied again — applying it twice is not the identity);
3. read or synthesize every annotator's masks (P1, P5, P3 x 10 annotators x
   15 patches) from the prototype's cache, the P3 gap under the declared
   stable seed;
4. compute every annotator's own C_a ONCE (Eq. 9) — the U1 unit's fixed
   values, which no rung recomputes;
5. measure the neighborhood's anchors once (random-mask floor, neutral-disk
   content floor, expert-pool consensus reference) and the calibration
   populations;
6. run BOTH ladders (P5 and P3) over the shared draw subsets, computing per
   draw: the U2 panel-consensus score, the supervised validation column, and
   every U1 arm's summary over every normalization;
7. reproduce the prototype's exact 6-draw protocol for the port-fidelity pass
   and diff it against the prototype's published CSV;
8. return one JSON-serializable record; the notebook renders it.

Nothing here selects a variant. The record is provisional and feeds the
operator's r10 deliberation.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from CrowdSegmenter_CT_GCE.metric import consistency, prototype
from CrowdSegmenter_CT_GCE.metric_pilot import (Anchors, MetricPilotConfig,
                                                SUMMARIES, VARIANTS,
                                                decision_metrics,
                                                early_detection,
                                                neutral_disk_mask,
                                                panel_consensus_score,
                                                panel_threshold,
                                                prototype_ladder_subsets,
                                                random_mask_masks,
                                                sample_ladder_subsets,
                                                summarize, u1_panel_scores)
from CrowdSegmenter_CT_GCE_Benchmark.metric_material import (
    AnnotatorPool, FeatureExtractor, evaluate_segmentation_batch, load_inria,
    majority_voting_consensus, resize_mask_to_features,
    soft_probability_consensus)


def run_metric_pilot(config: MetricPilotConfig) -> dict:
    """Run the whole pilot and return the JSON-serializable record."""
    record: dict = {
        "config": {k: getattr(config, k) for k in config.__dataclass_fields__},
        "status": "decided in r10 -- per-annotator unit, mean panel score with lower-decile early alarm, floor-anchored presentation",
        "revision": "research-concept-category-truncated-generalized-cross-entropy-for-semantic-se-r10.md",
        "timings": {},
    }
    t_start = time.time()

    images, gt_masks = load_inria(config.data_dir)
    image_ids = sorted(images)
    record["material"] = {
        "nPatches": len(image_ids),
        "patchIds": image_ids,
        "patchSize": list(images[image_ids[0]].size),
    }

    # -- features (the prototype's chain is baked into the extractor) --------
    t0 = time.time()
    extractor = FeatureExtractor()
    features = extractor.extract_features(images)
    record["timings"]["featureExtraction"] = time.time() - t0
    # sanity: the extractor's output is already spatially centered per channel
    for img in image_ids:
        residual = float(np.abs(features[img].mean(axis=(1, 2))).max())
        if residual > 1e-3:
            raise ValueError(f"features for {img} are not centered (residual {residual:.2e})")

    # -- annotator masks: cache reads for everything cached, stable seeds for
    #    the P3 gap -----------------------------------------------------------
    pool = AnnotatorPool(gt_masks, config.annotators_dir, config.base_seed)
    profiles = [config.good_profile, config.poor_profile, config.bias_profile]
    masks_256: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    masks_full: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    generated = 0
    for prof in profiles:
        for ann in range(config.n_annotators):
            m256: Dict[str, np.ndarray] = {}
            mfull: Dict[str, np.ndarray] = {}
            for img in image_ids:
                cache_path = pool.cache_dir / f"{prof}_ann{ann}_{img}.png"
                if not cache_path.exists():
                    generated += 1
                full = pool.generate_annotator_mask(img, prof, ann)
                mfull[img] = full
                m256[img] = resize_mask_to_features(full)
            masks_256[(prof, ann)] = m256
            masks_full[(prof, ann)] = mfull
    record["material"]["annotatorMasksGenerated"] = generated
    record["material"]["annotatorMasksReadFromCache"] = 3 * config.n_annotators * len(image_ids) - generated

    # -- per-annotator Eq. (9), computed ONCE (the U1 unit's fixed values) ---
    t0 = time.time()
    annotator_scores: Dict[Tuple[str, int], float] = {}
    for prof in profiles:
        for ann in range(config.n_annotators):
            protos = [prototype(features[img], masks_256[(prof, ann)][img]) for img in image_ids]
            c, _ = consistency(protos)
            annotator_scores[(prof, ann)] = c
    record["timings"]["perAnnotatorScores"] = time.time() - t0

    # -- the neighborhood's anchors, measured once, GT-free ------------------
    t0 = time.time()
    rng = np.random.RandomState(config.random_seed)
    random_masks = random_mask_masks(rng, 256, config.n_random_masks)
    floor_scores = []
    for mask in random_masks:
        protos = [prototype(features[img], mask) for img in image_ids]
        c, _ = consistency(protos)
        floor_scores.append(c)
    disk = neutral_disk_mask(256, config.disk_radius)
    protos = [prototype(features[img], disk) for img in image_ids]
    neutral_floor, _ = consistency(protos)
    ref_masks = {
        img: soft_probability_consensus(
            [masks_256[(config.good_profile, a)][img] for a in range(config.n_annotators)])
        for img in image_ids
    }
    protos = [prototype(features[img], ref_masks[img]) for img in image_ids]
    ceil_ref, _ = consistency(protos)
    anchors = Anchors(
        floor_rand=float(np.mean(floor_scores)),
        neutral_floor=float(neutral_floor),
        ceil_ref=float(ceil_ref),
        pop_good=[annotator_scores[(config.good_profile, a)] for a in range(config.n_annotators)],
        pop_poor=[annotator_scores[(config.poor_profile, a)] for a in range(config.n_annotators)],
    )
    record["timings"]["anchors"] = time.time() - t0
    record["anchors"] = {
        "floorRand": anchors.floor_rand,
        "floorRandPerMask": floor_scores,
        "neutralFloor": anchors.neutral_floor,
        "ceilRef": anchors.ceil_ref,
        "popGood": anchors.pop_good,
        "popPoor": anchors.pop_poor,
    }
    record["perAnnotatorScores"] = {
        f"{prof}|{ann}": c for (prof, ann), c in annotator_scores.items()
    }

    # -- the ladders over the shared subsets ----------------------------------
    subsets = sample_ladder_subsets(config.n_annotators, config.draws, config.subset_seed)
    record["drawSubsets"] = {str(c): [[list(g), list(b)] for g, b in pairs]
                             for c, pairs in subsets.items()}

    tau = {variant: panel_threshold(anchors, variant, config.threshold_k) for variant in VARIANTS}
    record["thresholds"] = tau

    ladders: Dict[str, dict] = {}
    for ladder_profile in (config.poor_profile, config.bias_profile):
        t_ladder = time.time()
        ladders[ladder_profile] = _run_ladder(
            config, image_ids, features, masks_256, masks_full, gt_masks,
            annotator_scores, anchors, tau, subsets, ladder_profile)
        record["timings"][f"ladder_{ladder_profile}"] = time.time() - t_ladder
    record["ladders"] = ladders

    # -- the U1 arm grid, assembled from the fixed annotator scores ----------
    for ladder_profile, ladder in ladders.items():
        for variant in VARIANTS:
            normalized = {
                int(c): anchors.normalized(np.asarray(ladder["u1RawByDraw"][c]), variant)
                for c in sorted(ladder["u1RawByDraw"], key=int)
            }
            ladder["arms"][variant] = {}
            for summary in SUMMARIES:
                rungs = [np.array([summarize(list(row), summary, tau[variant], config.quantile)
                                   for row in normalized[c]])
                         for c in sorted(normalized)]
                arm = decision_metrics(rungs, ladder["supervisedMeanByRung"])
                arm["earlyDetection"] = early_detection(rungs, upto=3)
                ladder["arms"][variant][summary] = arm

    # -- the port-fidelity pass: the prototype's exact protocol --------------
    record["portFidelity"] = _port_fidelity(
        config, image_ids, features, masks_256, gt_masks)

    record["timings"]["total"] = time.time() - t_start
    return record


def _run_ladder(config: MetricPilotConfig, image_ids: List[str], features: Dict[str, np.ndarray],
                masks_256: Dict[Tuple[str, int], Dict[str, np.ndarray]],
                masks_full: Dict[Tuple[str, int], Dict[str, np.ndarray]],
                gt_masks: Dict[str, np.ndarray],
                annotator_scores: Dict[Tuple[str, int], float],
                anchors: Anchors, tau: Dict[str, float],
                subsets: Dict[int, List[Tuple[Tuple[int, ...], Tuple[int, ...]]]],
                ladder_profile: str) -> dict:
    """One contaminant profile's ladder: U2 per draw, supervised per draw, U1 assembled."""
    ladder: dict = {
        "profiles": [config.good_profile, ladder_profile],
        "u2ByDraw": {},
        "u1RawByDraw": {},
        "supervisedByDraw": {},
        "arms": {},
    }
    supervised_means: List[float] = []
    for c in sorted(subsets):
        u2_rows, sup_rows, u1_rows = [], [], []
        for good_ids, bad_ids in subsets[c]:
            u2_protos: Dict[str, np.ndarray] = {}
            consensus_bin: Dict[str, np.ndarray] = {}
            for img in image_ids:
                panel_256 = ([masks_256[(config.good_profile, a)][img] for a in good_ids]
                             + [masks_256[(ladder_profile, a)][img] for a in bad_ids])
                consensus = soft_probability_consensus(panel_256)
                u2_protos[img] = prototype(features[img], consensus)
                panel_full = ([masks_full[(config.good_profile, a)][img] for a in good_ids]
                              + [masks_full[(ladder_profile, a)][img] for a in bad_ids])
                consensus_bin[img] = majority_voting_consensus(panel_full)
            u2 = panel_consensus_score(u2_protos, image_ids)
            sup = evaluate_segmentation_batch(consensus_bin, gt_masks)
            panel_annotators = ([("P1", a) for a in good_ids]
                                + [(ladder_profile, a) for a in bad_ids])
            u1_rows.append([annotator_scores[key] for key in panel_annotators])
            u2_rows.append(u2)
            sup_rows.append({"iou": sup["iou"], "miou": sup["miou"], "dice": sup["dice"]})
        ladder["u2ByDraw"][str(c)] = u2_rows
        ladder["u1RawByDraw"][str(c)] = u1_rows
        ladder["supervisedByDraw"][str(c)] = sup_rows
        supervised_means.append(float(np.mean([r["miou"] for r in sup_rows])))
    ladder["supervisedMeanByRung"] = supervised_means
    u2_arm = decision_metrics(
        [np.array(ladder["u2ByDraw"][str(c)]) for c in sorted(subsets)], supervised_means)
    u2_arm["earlyDetection"] = early_detection(
        [np.array(ladder["u2ByDraw"][str(c)]) for c in sorted(subsets)], upto=3)
    ladder["u2Arm"] = u2_arm
    return ladder


def _port_fidelity(config: MetricPilotConfig, image_ids: List[str],
                   features: Dict[str, np.ndarray],
                   masks_256: Dict[Tuple[str, int], Dict[str, np.ndarray]],
                   gt_masks: Dict[str, np.ndarray]) -> dict:
    """The prototype's exact 6-draw protocol, reproduced and diffed against its CSV.

    The compositions are the prototype's own (same seed, same sampling
    sequence), so any difference is device/precision drift of the port, not
    sampling. The pass covers U2 (the arm the prototype measured); the U1 unit
    has no published counterpart.
    """
    subsets = prototype_ladder_subsets(config.n_annotators, config.subset_seed)
    reference = _load_prototype_csv(config.prototype_csv)
    rows = []
    for c in sorted(subsets):
        draws = []
        for good_ids, bad_ids in subsets[c]:
            draws.append(_port_fidelity_draw(
                config, image_ids, features, masks_256, good_ids, bad_ids))
        ref = reference.get(str(c))
        rows.append({
            "c": c,
            "nDraws": len(draws),
            "portMean": float(np.mean(draws)),
            "prototypeCk": ref,
            "absDiff": None if ref is None else float(abs(np.mean(draws) - ref)),
        })
    diffs = [r["absDiff"] for r in rows if r["absDiff"] is not None]
    return {
        "protocol": "the prototype's own: 6 draws per rung, Random(42), its exact sampling sequence",
        "rows": rows,
        "maxAbsDiff": max(diffs) if diffs else None,
        "tolerance": 1e-3,
        "note": "the prototype's CSV was produced on an unknown device; the "
                "port runs on CPU, so small precision drift is expected and "
                "reported per rung",
    }


def _port_fidelity_draw(config: MetricPilotConfig, image_ids: List[str],
                        features: Dict[str, np.ndarray],
                        masks_256: Dict[Tuple[str, int], Dict[str, np.ndarray]],
                        good_ids: Tuple[int, ...], bad_ids: Tuple[int, ...]) -> float:
    """One 6-draw-protocol panel's U2 score (the prototype's exact compositions)."""
    protos = {}
    for img in image_ids:
        panel_256 = ([masks_256[(config.good_profile, a)][img] for a in good_ids]
                     + [masks_256[(config.poor_profile, a)][img] for a in bad_ids])
        consensus = soft_probability_consensus(panel_256)
        protos[img] = prototype(features[img], consensus)
    return panel_consensus_score(protos, image_ids)


def _load_prototype_csv(path: str) -> Dict[str, float]:
    """The prototype's published contamination table: {c: ck}."""
    out: Dict[str, float] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[str(int(row["c"]))] = float(row["ck"])
    return out
