"""Reference implementation of the CrowdSegmenter-CT-GCE formulation.

Each module declares the sections and equations it implements in
`__provenance__`, and every invariant listed there has a matching
test under tests/.
"""

from . import (bound_study, dawid_skene, identifiability, metric, metric_pilot,
               module, reliability, selection_study, thresholds)
from .bound_study import (CrowdModel, bayes_risk, compare_cutoffs,
                          consensus_posterior, expected_truncated_loss,
                          noise_rates, population_rho)
from .dawid_skene import (fit_dawid_skene, majority_disagreement,
                          majority_posterior, rho_hat)
from .identifiability import (CANDIDATES, c1_penalty, c2_loss, c3_loss,
                              fit_reliability_branch,
                              flip_rates_from_confusions,
                              likelihood_integrand)
from .metric import (boundary_ring, calibration_threshold, consistency,
                     fraction_below, panel_lower_quantile, panel_mean,
                     panel_median, prepare_features, prototype)
from .metric_pilot import (Anchors, MetricPilotConfig, VARIANTS, SUMMARIES,
                           decision_metrics, early_detection,
                           panel_consensus_score, panel_threshold,
                           prototype_ladder_subsets, sample_ladder_subsets,
                           summarize, u1_panel_scores, verdict_for_step)
from .module import (ctgce_grad_lambda, ctgce_grad_p, ctgce_integrand,
                     ctgce_loss)
from .reliability import (IdentifiabilityConstraintRequired, annotated_labels,
                          gather_at_annotated,
                          require_identifiability_constraint,
                          unconstrained_optimum, validate_reliability)
from .selection_study import (LAMBDA_GRID, SCENARIOS, StudyConfig,
                              collapse_diagnosis, floor_sensor,
                              gate_correlation, generate_world,
                              pruning_precision_recall, recovery_error,
                              run_cell, true_reliability_table,
                              truth_gate_correlation)
from .thresholds import (Calibration, calibrate_cutoffs, category_confidences,
                         empirical_quantile)

__all__ = [
    "bound_study",
    "dawid_skene",
    "identifiability",
    "metric",
    "metric_pilot",
    "module",
    "reliability",
    "selection_study",
    "thresholds",
    "CrowdModel",
    "bayes_risk",
    "compare_cutoffs",
    "consensus_posterior",
    "expected_truncated_loss",
    "noise_rates",
    "population_rho",
    "fit_dawid_skene",
    "majority_disagreement",
    "majority_posterior",
    "rho_hat",
    "CANDIDATES",
    "c1_penalty",
    "c2_loss",
    "c3_loss",
    "fit_reliability_branch",
    "flip_rates_from_confusions",
    "likelihood_integrand",
    "ctgce_grad_lambda",
    "ctgce_grad_p",
    "ctgce_integrand",
    "ctgce_loss",
    "IdentifiabilityConstraintRequired",
    "annotated_labels",
    "gather_at_annotated",
    "require_identifiability_constraint",
    "unconstrained_optimum",
    "validate_reliability",
    "LAMBDA_GRID",
    "SCENARIOS",
    "StudyConfig",
    "collapse_diagnosis",
    "floor_sensor",
    "gate_correlation",
    "generate_world",
    "pruning_precision_recall",
    "recovery_error",
    "run_cell",
    "true_reliability_table",
    "truth_gate_correlation",
    "Calibration",
    "calibrate_cutoffs",
    "category_confidences",
    "empirical_quantile",
    "boundary_ring",
    "calibration_threshold",
    "consistency",
    "fraction_below",
    "panel_lower_quantile",
    "panel_mean",
    "panel_median",
    "prepare_features",
    "prototype",
    "Anchors",
    "MetricPilotConfig",
    "VARIANTS",
    "SUMMARIES",
    "decision_metrics",
    "early_detection",
    "panel_consensus_score",
    "panel_threshold",
    "prototype_ladder_subsets",
    "sample_ladder_subsets",
    "summarize",
    "u1_panel_scores",
    "verdict_for_step",
]

__implementation__ = {
    # The managed revision this declaration is bound to -- the filename the
    # deliberation engine's STATUS reports as latest, confirmed by the
    # operator at the object-map gate.
    "revision": "research-concept-category-truncated-generalized-cross-entropy-for-semantic-se-r10.md",

    # Carried unchanged from r03's secondary claim: consensus segmentation,
    # judged per pixel against ground truth by mIoU and Dice, higher better.
    "premises": {
        "prediction": "a class label per pixel (consensus segmentation)",
        "statisticalUnit": "pixel",
        "metric": "mIoU and Dice",
        "direction": "higher",

        # The synthetic selection pilot r05 § "Falsifiable claim and
        # assumptions" declares ("Constraint selection (synthetic study)"),
        # with its exact declared configuration. r10 closed the selection:
        # the coupled likelihood alone (Ec. 7, C3) is the constraint and the
        # soft gating of (Ec. 1) is the design, for the reasons r10 §
        # "Constraint selection (synthetic study)" states. The configuration
        # below is the record of what was run.
        "syntheticSelectionPilot": {
            "status": "decided in r10 -- C3 (Ec. 7) is the constraint, soft gating of (Ec. 1) is the design",
            "candidates": {
                "C1": "Eq. (5): mean-reliability normalization against the "
                      "EM's per-annotator per-category budget 1 - rho_{r,k}, "
                      "with rho_{r,k} = 1 - T_hat_r[k,k]",
                "C2": "Eq. (6): the coupled symmetric-noise likelihood plus "
                      "Zhang (2023)'s fidelity-plus-trace penalty, weight "
                      "lambda",
                "C3": "Eq. (7): the coupled symmetric-noise likelihood alone",
            },
            "trainingSignal": "phi is trained against the candidate's own "
                              "objective with the segmentation held fixed, "
                              "never by (Ec. 1)",
            "branch": "a per-(annotator, category, region) logit table, "
                      "sigmoid output, zero init -- the smallest branch that "
                      "can express the declared truths, so capacity is not "
                      "the confound",
            "optimizer": {"name": "Adam (full batch)", "lr": 0.05,
                          "steps": 1500},
            "world": {
                "K": 4, "R": 4, "regions": ["mass", "boundary"],
                "N": 4000, "coverage": 0.8,
                "prior": [0.4, 0.3, 0.2, 0.1],
                "regionSplit": [0.7, 0.3],
                "segmentationProxy": {"perRegionModelError": [0.10, 0.30],
                                      "softness": 0.15},
            },
            "scenarios": {
                "region-varying": {
                    "accuracyMass": [0.95, 0.90, 0.85, 0.80],
                    "accuracyBoundary": [0.65, 0.60, 0.55, 0.50],
                    "whatVaries": "region"},
                "per-category": {
                    "accuracyByCategory": [0.95, 0.80, 0.60, 0.40],
                    "whatVaries": "category"},
                "correlated-bias": {
                    "accuracy": 0.75,
                    "mechanism": "one shared wrong category per pixel: every "
                                 "annotator who flips there writes it (A1 "
                                 "violated, marginals unchanged)",
                    "whatVaries": "the error correlation"},
            },
            "lambdaGridC2": [0.05, 0.2, 0.8, 3.2],
            "seeds": {"pilot": 5, "verdictFloor": 30,
                      "note": "the pilot runs 5 seeds and therefore grants "
                              "no verdict; the deliberation declares the "
                              "campaign scale"},
            "metrics": {
                "recoveryError": "mean |Lambda_trained - Lambda_true| over "
                                 "annotated entries at the annotated slot",
                "gateCorrelation": "Pearson correlation between "
                                   "Lambda_trained and the truncation gate "
                                   "1[P[n,k_r] > C_tilde_{k_r}]; the "
                                   "unconstrained reference of Eq. (2) is "
                                   "the gate itself (correlation 1) and the "
                                   "truth's own correlation is reported "
                                   "beside it",
                "collapse": "a run collapses when >= 90% of its evaluated "
                            "entries sit within 0.02 of 0 or 1, or the "
                            "field's standard deviation is < 0.01; the rate "
                            "is the fraction of seeds that collapse",
                "pruning": "per category, prune the ceil(rho_hat_k * count_k) "
                           "lowest-reliability annotations (stable (r, n) "
                           "order) and score precision/recall against the "
                           "annotations the generator knows are wrong",
            },
            "truth": "the generator's closed-form agreement probability "
                     "given the ANNOTATED category (r05's own definition of "
                     "Lambda); the generator's correctness table is recorded "
                     "beside it",
            "c1Budget": "rho_{r,k} = 1 - T_hat_r[k,k] from the EM confusion "
                        "matrices, exactly as Eq. (5) derives it",
            "sensor": "F_k(1/K) per category after the warm-up -- how much "
                      "the 1/K clip discards beyond rho_hat_k, and where",
        },

        # The round-2 design comparison the operator commissioned: three STUDY
        # arms derived from Eq. (1), not the production loss. r10 decided it:
        # the soft gating of (Ec. 1) is preferred over the hard-gate variants
        # (S1/S2), which prune hard-but-clean pixels and freeze their gradient.
        # The world, scenarios and seeds reproduce round 1 exactly
        # (same generator, same deterministic draw per (scenario, seed)), so the
        # tables are comparable; C_tilde_k are round 1's (Eq. 4 over the same
        # warm-up proxy), and phi is trained by C3 (Ec. 7) with the segmentation
        # held fixed -- the round-1 protocol, identical across arms.
        "syntheticDesignRound": {
            "status": "decided in r10 -- soft gating of (Ec. 1) is the design",
            "arms": {
                "S1": "no Lambda: the gate prunes and surviving annotations "
                      "weigh 1 -- Eq. (1) with Lambda replaced by the analytic "
                      "indicator 1{P > C_tilde_k} (the thesis' truncation)",
                "S2": "gate prunes + Lambda weighs: the hard prune decision is "
                      "1{P > C_tilde_k}; Lambda multiplies the GCE term of the "
                      "survivors, as a soft weight",
                "S3": "Lambda prunes: Eq. (1) exactly as written today (Lambda "
                      "decides between the GCE term and the floor)",
            },
            "sharedByEveryArm": {
                "reliability": "phi = the per-(annotator, category, region) "
                               "logit table, trained by C3 (Ec. 7) against the "
                               "FIXED warm-up proxy P -- the round-1 protocol, "
                               "identical for every arm; the Lambda metrics are "
                               "therefore arm-shared by construction, which is "
                               "the control the design holds",
                "cutoffs": "C_tilde_k from Eq. (4) over the warm-up proxy, the "
                           "round-1 values",
                "world": "round 1's generator, bitwise (truth, annotations, "
                         "mask, P proxy) per (scenario, seed)",
            },
            "segmenter": {
                "features": "2-D per-pixel features from class-conditional "
                            "Gaussians on the unit circle (means (1,0), (0,1), "
                            "(-1,0), (0,-1), unit isotropic covariance), drawn "
                            "from their own generator seeded with the run's "
                            "seed",
                "model": "linear softmax over the features, zero init",
                "schedule": "A3 warm-up first: 300 steps of plain crowd GCE "
                            "(no truncation, identical for every arm), then "
                            "1200 steps of the arm's loss; Adam lr 0.05, full "
                            "batch, q = 0.7; the gate and Lambda are constants "
                            "inside each segmentation step (phi never trains by "
                            "Eq. 1)",
            },
            "primaryMetric": "consensus segmentation quality against the "
                             "synthetic truth: mIoU and Dice of argmax P "
                             "against the consensus labels (mIoU excludes "
                             "classes absent from both sides; Dice scores 1.0 "
                             "when both are empty), and the error of P against "
                             "the true probabilities -- the true field is the "
                             "point mass on the consensus, so the declared "
                             "error is cross-entropy -mean log P[n, k*]",
            "secondaryMetrics": "the round-1 four, read per arm through the "
                                "arm's own mechanism: recovery/gate-correlation/"
                                "collapse on the shared C3 Lambda; pruning "
                                "precision/recall of each arm's OWN hard prune "
                                "(S1/S2 prune where the gate is 0; S3 prunes "
                                "where Lambda < 0.5, the point where Eq. 1's "
                                "mixture hands the weight from GCE to floor) "
                                "against the annotations the generator knows "
                                "are wrong",
            "survivingLambda": "for S2 and S3: the mean Lambda over each arm's "
                               "own survivors per scenario and category (S2: "
                               "the gate's survivors; S3: Lambda >= 0.5) -- "
                               "the statistic that will later decide whether "
                               "the weight needs normalization",
            "sensor": "F_k(1/K) per category after the warm-up (round-1 "
                      "definition, world-level, arm-shared)",
            "seeds": {"target": 30, "verdictFloor": 30,
                      "round1Comparable": "seeds 0..4 reproduce round 1's "
                                          "worlds bitwise",
                      "note": "if the measured cost forbids 30, run at least "
                              "15 and say so clearly"},
        },

        # The metric pilot (r10 § "The evaluation metric: neighborhood latent
        # consistency" and the Metric claim paragraph): the per-annotator unit
        # of Eqs. (8)-(9) against the prototype's panel-consensus variant, over
        # the prototype's own contamination ladder. r10 closed every selection:
        # the per-annotator unit over the panel-consensus variant, the mean of
        # {C_a} as panel score with the lower decile as early alarm, and
        # floor-anchored presentation. The zero-GT decision metrics never see
        # the supervised column, which validates only.
        "metricPilot": {
            "status": "decided in r10 -- per-annotator unit, mean panel score "
                      "with lower-decile early alarm, floor-anchored presentation",
            "binding": "r10 Eqs. (8)-(9), per-annotator, no ground truth; the "
                       "panel-consensus variant of the first prototype is the "
                       "comparison arm",
            "material": {
                "data": "the prototype's 15 Inria patches (1024x1024) + GT "
                        "masks, copied verbatim into data/images and "
                        "data/masks (23 MB with the annotator cache)",
                "annotatorCache": "data/annotators/: the prototype's 601 "
                                  "cached mask PNGs, read verbatim -- the "
                                  "prototype seeds generation with Python's "
                                  "process-randomized hash(), so the cache "
                                  "is the ONLY stable source for P1/P2/P4/P5; "
                                  "regenerating a cached profile would NOT "
                                  "reproduce the prototype's masks",
                "p3masks": "P3 ann0..9 x 15 patches: 1 existed in the cache; "
                           "the other 149 are generated fresh under "
                           "stable_seed = base_seed + crc32(image_id) % "
                           "100000 + ann*1000 + profile chars -- declared, "
                           "process-stable, never bit-comparable to anything "
                           "earlier because the prototype never ran a P3 "
                           "ladder",
                "backbone": "torchvision ResNet-34 ImageNet weights "
                            "(resnet34-b627a593.pth, ~84 MB), downloaded once "
                            "with the operator's approval, read from the torch "
                            "hub cache; features = layer2+layer3, both "
                            "bilinearly upsampled to 256x256, concatenated to "
                            "384 channels, then the prototype's prepare chain "
                            "(metric.prepare_features)",
                "device": "CPU, declared -- bitwise reproducibility; every arm "
                          "measured on the same device",
            },
            "neighborhood": "N = all 15 patches (the prototype's own "
                            "neighborhood); |N| = 15 satisfies Eq. (9)'s "
                            "|N_a| >= 2, and the metric module fails closed "
                            "below it",
            "seeds": {"base": 42, "subsetSampling": 42, "randomFloor": 7,
                      "drawsPerRung": 30, "verdictFloor": 30,
                      "prototypeFidelityPass": "the prototype's own protocol "
                                               "(6 draws, Random(42), its "
                                               "exact sampling sequence) "
                                               "reproduced verbatim; its "
                                               "published CSV is the "
                                               "comparison table "
                                               "(data/prototype_reference/)",
                      "note": "rungs c=0 and c=10 have exactly one panel "
                              "composition, so their scores are deterministic "
                              "and carry no dispersion"},
            "ladder": {
                "P5arm": "rung c = (10-c) P1 + c P5, c = 0..10 -- the "
                         "prototype's bench",
                "P3arm": "rung c = (10-c) P1 + c P3, same subsets -- the "
                         "declared scope measurement: consistency is not "
                         "correctness, reported apart, never hidden",
                "drawSubsets": "one sample per ladder, fixed by seed 42 BEFORE "
                               "any arm runs and reused by every unit, "
                               "normalization and summary (arms differ in what "
                               "they compute, never in what they touch)",
            },
            "units": {
                "U1": "per-annotator, Eq. (9): C_a over the 15 patches per "
                      "annotator; the panel score summarizes {C_a}",
                "U2": "panel consensus (the prototype's original): the soft "
                      "probability consensus of the panel per patch, then one "
                      "consistency over the neighborhood -- the compressed "
                      "ladder 0.7984->0.7865 is its published record",
            },
            "normalizations": {
                "N0": "raw C_a",
                "S1": "(C_a - floor_rand) / (ceil_ref - floor_rand); floor = "
                      "mean consistency of 32 smooth random masks (seed 7), "
                      "ceiling = the consistency of the full expert pool's "
                      "(10 P1) soft-consensus reference masks; anchors fixed "
                      "once, GT-free, panel-independent",
                "S2z": "(C_a - mu_good) / sigma_good against the 10-expert "
                       "calibration population measured on the same "
                       "neighborhood; declared limitation: needs a calibration "
                       "cohort",
                "S2p": "the score's percentile within the pooled 20-annotator "
                       "population (10 P1 + 10 P5) measured once on the same "
                       "neighborhood; same limitation",
                "S3": "(C_a - neutral_floor) / (ceil_ref - neutral_floor); "
                      "neutral floor = consistency of prototypes under one "
                      "fixed centered disk (r = 64 at 256), the content term "
                      "any annotation inherits from the patches being similar",
            },
            "panelSummaries": {
                "mean": "arithmetic mean of the panel's normalized scores",
                "median": "median",
                "q10": "lower decile (quantile 0.1, linear interpolation)",
                "fracBelow": "fraction of annotators below tau = mu_good - 1 "
                             "sigma_good, computed in each variant's own "
                             "scale; direction OPPOSITE to the others "
                             "(grows with contamination) and declared per row",
            },
            "decisionMetrics": "zero-GT: dynamicRange score(0)-score(10); "
                               "monotonic violations raw and beyond 2 combined "
                               "SEs; adjacent-step separation |delta|/SE_comb "
                               "(min over steps and the c=1->2 step, the 9+1 "
                               "vs 8+2 question); early detection over "
                               "c <= 3 per summary; P3 arm apart. The "
                               "supervised column (majority vote at 1024, "
                               "IoU/mIoU/Dice vs the dataset GT) validates "
                               "only: rank correlation reported, never a "
                               "selection input (invariant "
                               "pilot_supervised_column_validation_only)",
            "verdictRule": "no verdict below 30 draws per rung or within 2 "
                           "combined SEs; endpoints are deterministic and "
                           "carry no dispersion; every no-verdict carries its "
                           "reason",
            "openAfterThePilot": "the selection itself (r10); whether S2's "
                                 "calibration cohort can be replaced by a "
                                 "self-calibrating population; whether the "
                                 "neighborhood reference should move with the "
                                 "measured panel (S1/S3's ceiling is fixed at "
                                 "the expert-pool consensus here)",
        },
    },

    # This repository keeps its product tree at the repository root: notebooks/,
    # results/, models/, AGREED.md and .implementation/ sit beside src/ and
    # tests/, with no `<Name>/` folder repeating the repository's own name.
    # `product_root` in the implementation engine reads this key; an absent key
    # keeps the default `<target>/<Name>/` layout for every other target.
    "productRoot": ".",
}

# The ordered ladder of rungs a position-section step can reach, entirely
# in this repository's own words -- the forge holds no rung name of its
# own, only the arithmetic that compares two of these names by position
# (see `impl_position.level_index`). A step earns a rung by naming this
# file's own ladder explicitly on its witness (`` `@rehearsal:level <job>`
# `` in `AGREED.md`'s position section); a step with no `:level` marker is
# two-state and never reads this list at all. Left empty until named -- a
# repository whose position items are entirely two-state needs no ladder
# here, and one is never invented on its behalf. A second, independent
# top-level literal, held apart from `__implementation__` above for the
# same reason `resolve_levels_declaration`'s own docstring states.
#
# Example (a repository with no remote service at all still has a ladder):
#     __levels__ = ["local", "cluster"]
__levels__: list = []

# A callable this repository's own code can run, isolated, under this
# repository's own venv -- named and resolved statically by the forge
# (module + function, never imported here), then imported and called
# inside the target's own interpreter, never the forge's. A third,
# independent top-level literal, held apart from `__implementation__` for
# the identical reason `__levels__` is: see `resolve_steps_declaration`'s
# own docstring. Left empty until a step exists -- a repository with
# nothing local to run in isolation needs none, and one is never invented
# on its behalf.
#
# Each entry carries `module` and `function`, and one further key that is
# asked of every step and defaulted for none: `produces`, the list of
# path roots -- relative to the product folder -- that this step and only
# this step writes into. `step` snapshots the product folder before and
# after every run and reports what changed on each side of those roots,
# which is the only way it can tell a step that returned having written
# nothing from one that produced its whole output, or a step that stayed
# in its own tree from one that wrote into a neighbour's. Leave it out
# and BOTH readings are switched off for that step: `verify` says so, per
# step, in `undeclaredProduces`, and nothing here defaults a root on your
# behalf -- the forge never guesses which work belongs to which step.
#
# The pattern that key exists to make possible, and the one worth
# scaffolding on the first day rather than discovering after a run: a
# flow has two kinds of step, and each one owns a notebook.
#
#   - a step that COMPUTES -- it orchestrates this package's own
#     library, writes data, and draws nothing;
#   - a step that DRAWS -- it reads what the computing step left on
#     disk and renders tables, figures and conclusions.
#
# Each names its own notebook among its `produces` roots, so a pilot
# executes both of them AS notebooks. That is the point of the split:
# the artefact that is later handed to a worker elsewhere is a
# notebook, so a pilot that exercises anything else has not tested
# what gets sent.
#
# Collapsing the two into one step is a legitimate design and nothing
# in this forge refuses it -- but what it costs is written down here
# so the choice is made rather than defaulted into. A figure can no
# longer be redrawn without paying for the computation behind it
# again; and whichever half is left outside a notebook is the half
# the pilot never exercised in the shape it will be sent in. `verify`
# says so, per step, in `undeclaredStepNotebooks`, and it never
# refuses -- it names what the absence costs and leaves the design
# yours.
#
# And one more key, asked of every step and defaulted for none:
# `placement`, which says WHERE the step runs once the flow leaves
# rehearsal scale -- "local" on this machine, or "remote" on a worker.
# A remote one also names two more: the `job` folder that carries it
# and the `service` that folder lives under. Nothing else ties a step
# to a job, and the forge deliberately does not invent that link --
# which work goes through which job folder is this repository's
# layout, not the forge's to guess. The service is yours to name for
# a harder reason: the forge may read a service name to walk a
# directory and must reduce it to a count before returning anything,
# so nothing there can name one, and it cannot discover one either --
# adapters register lazily, so the registry is empty until somebody
# names one.
#
# It is a DECLARATION and not a decision recorded somewhere else, and
# the reason is worth having on the first day. That routing gets
# decided in conversation, and a conversation lands in the ledger
# under `.implementation/` -- free prose, in a directory `.gitignore`
# excludes. So a decision left there can be neither consumed (nothing
# parses a sentence into a route) nor travelled with (a clone
# receives none of it), while the walk that has to act on it runs
# from a clone. The ledger keeps the REASON, with its numbers; this
# key carries the FACT, which is the half a machine reads.
#
# Leave it out and the walk cannot route that step: it knows the step
# exists, what it produces and where it sits in the order, and not
# whether it runs here or elsewhere. `verify` says so, per step, in
# `undeclaredPlacement`, and it never refuses -- but nothing defaults
# it either. Routing an unrouted step by convention is how a run
# measured in days lands somewhere nobody chose, so `probe`'s own
# `flowActs` reports that step as `blocked` and names what is missing
# rather than picking for you.
#
# Example -- two steps, each owning its own notebook, each saying
# where it runs:
#     __steps__ = {
#         "computation": {
#             "module": "Example_Method.steps",
#             "function": "run_computation",
#             # Where this step sits in the order, and what it
#             # consumes from the steps above it. `advances` is the
#             # position item this step produces evidence for;
#             # `reads` is empty here because nothing precedes it,
#             # and an empty list is an answer -- it is what tells a
#             # remote rehearsal there is no upstream output to wait
#             # for.
#             "advances": 1,
#             "reads": [],
#             # Writes data and draws nothing. Its notebook is what
#             # the pilot executes and what a worker elsewhere would
#             # be handed.
#             "produces": ["Results/computation",
#                          "Notebooks/computation.ipynb"],
#             # The expensive half, so it goes to a worker at full
#             # scale -- and it names the job folder that carries it
#             # there.
#             "placement": "remote",
#             "job": "computation",
#             "service": "the-service-you-send-to",
#         },
#         "rendering": {
#             "module": "Example_Method.steps",
#             "function": "run_rendering",
#             "advances": 2,
#             # What it consumes, named: this is the link a remote
#             # rehearsal reads to refuse before opening a notebook
#             # whose inputs are not there yet, naming each missing
#             # root and which step writes it.
#             "reads": ["Results/computation"],
#             # Reads what "computation" left behind and renders. It
#             # writes no data of its own, so its notebook is the
#             # whole of what it produces -- and it can be re-run on
#             # its own, without paying for the computation again.
#             "produces": ["Notebooks/rendering.ipynb"],
#             # Seconds of drawing, and the artefact a person reads.
#             # Sending it to a worker would put the thing somebody
#             # has to read behind a download and buy nothing, so it
#             # stays here and names no job.
#             "placement": "local",
#         },
#     }
__steps__: dict = {}

# A target-chosen name mapped to the record it addresses -- a leveled
# `@record:level <name>` witness in AGREED.md's position section reaches
# exactly one entry here, deriving its rung through the identical
# arithmetic the benchmark's own `search` block already uses
# (`impl_position._record_scale_level`). A fourth, independent top-level
# literal, held apart from `__implementation__` for the identical reason
# `__levels__`/`__steps__` are: see `resolve_records_declaration`'s own
# docstring. Left empty until a record is named -- a repository whose
# leveled `@record` witness stays the bare, operand-less form needs no
# entry here, and one is never invented on its behalf.
#
# Example:
#     __records__ = {
#         "main": {
#             "path": "results/record.json",
#             "requiredScale": {"seeds": 3},
#         },
#     }
__records__: dict = {}
