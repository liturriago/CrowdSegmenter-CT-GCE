"""The round-2 design-comparison arms: three study forms derived from Eq. (1).

r05 § "The loss" states Eq. (1) and the identifiability doctrine; § "Open
items" keeps the design open. This module implements the round-2 synthetic
comparison the operator commissioned: THREE STUDY ARMS derived from Eq. (1) —
not the production loss, and no arm is the design until the deliberation
publishes r06. All three arms share the C3-trained reliability (Ec. 7, the
round-1 selection study's best-recovering candidate) and the same Eq. (4)
cutoffs; they differ in exactly one thing, the loss through which the
segmentation trains:

- S1 — no Λ: the gate prunes; surviving annotations weigh 1. Eq. (1) with Λ
  replaced by the analytic indicator 1{P > C̃_k}: the thesis' truncation, the
  prior-art boundary of r05.
- S2 — gate prunes + Λ weighs: the hard prune decision is the gate; Λ
  multiplies the GCE term of the survivors, as a soft weight.
- S3 — Λ prunes: Eq. (1) exactly as written today (Λ decides between the GCE
  term and the floor). `arm_integrand("S3", ...)` is asserted equal to
  `module.ctgce_integrand` — the arm IS the proposal's loss.

The arm integrand, one shape for all three (masked mean over every (r, w, h)
entry, the convention Eq. (1) fixes):

    integrand = M · ( w · GCE(P[k_r]) + (1 − w) · floor(C̃_{k_r}) ),

with GCE(p) = (1 − p^q)/q, floor(c) = (1 − c^q)/q, and the arm's weight
w = 1{P > C̃} (S1), g·Λ (S2), or Λ (S3). The gate and Λ are constants inside a
segmentation step (φ never trains by Eq. 1 — r05's doctrine), so the floor
term contributes no gradient and the GCE gradient of every entry is scaled by
its arm weight.

**The round-2 world.** Round 1's generator is reused unchanged — same truth,
annotations, mask and fixed P proxy per (scenario, seed) (`generate_world` is
called, not copied; the comparability invariant
`design_world_shares_round_one_world` asserts the shared tensors bitwise
equal). What round 2 adds is a declared trainable segmentation problem: 2-D
per-pixel features from class-conditional Gaussians on the unit circle
(means (1,0), (0,1), (−1,0), (0,−1), unit isotropic covariance), drawn from
their own generator seeded with the run's seed. The segmenter is a linear
softmax over the features, trained full-batch in two declared phases
(r05 A3: truncation is enabled only after the warm-up):

1. a warm-up phase, identical for every arm: plain crowd GCE over all
   annotated entries — no truncation — so the gate opens on a live
   confidence distribution;
2. the arm phase: the arm's integrand above, gate and Λ held constant
   inside each step.

**Metrics.** Primary (never measured before): the consensus quality of the
trained P against the synthetic truth — mIoU and Dice of argmax P against the
consensus labels (mIoU excludes classes absent from both sides; Dice scores
1.0 when both sides are empty), and the error of P against the true
probabilities — the true probability field is the point mass on the consensus,
so the declared error is the cross-entropy −mean log P[n, k*]. Secondary (the
round-1 four, read per arm through the arm's own mechanism): the Λ metrics
(recovery, gate correlation, collapse) are computed once per (scenario, seed)
on the shared C3-trained reliability — they are arm-shared by construction,
which is the control the design holds; the pruning precision/recall is each
arm's OWN hard prune against the annotations the generator knows are wrong:
S1/S2 prune where the gate is 0 (their mechanism), S3 prunes where Λ < 0.5
(the point where Eq. 1's mixture hands the weight from GCE to floor). The
surviving-Λ mean is the mean reliability over each arm's own survivors, per
scenario and category — S2 over the gate's survivors, S3 over Λ ≥ 0.5 — the
statistic that will later decide whether the weight needs normalization.

One module per mathematical object; `__provenance__` is read statically by
`implementation_cli.py verify` — the module is never imported to obtain it —
and it is the only thing that makes revision drift detectable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .module import ctgce_integrand
from .reliability import annotated_labels, gather_at_annotated
from .selection_study import (BRANCH_LR, BRANCH_STEPS, UNCONSTRAINED_Q,
                              StudyConfig, collapse_diagnosis,
                              estimate_thresholds, floor_sensor,
                              gate_correlation, generate_world, recovery_error,
                              train_candidate)

__provenance__ = {
    "revision": "research-concept-category-truncated-generalized-cross-entropy-for-semantic-se-r10.md",
    "sections": ["The loss"],
    "equations": ["1"],
    "invariants": [
        "s3_arm_is_eq1",
        "s1_truncation_gradient_vanishes_on_pruned",
        "s2_survivor_weight_is_gate_times_lambda",
        "consensus_metrics_match_manual",
        "survivor_sets_match_manual",
        "design_world_shares_round_one_world",
        "features_deterministic_per_seed",
    ],
}

#: The arms' names, and the only ones every entry point accepts.
ARMS = ("S1", "S2", "S3")

#: q for the GCE and the floor, shared with the round-1 reference.
DESIGN_Q = UNCONSTRAINED_Q

#: The declared feature space: class-conditional Gaussians on the unit circle.
FEATURE_MEANS = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))

#: The segmenter's declared training schedule (r05 A3: warm-up first).
THETA_WARMUP_STEPS = 300
THETA_ARM_STEPS = 1200
THETA_LR = 0.05

#: Λ's handover point in Eq. 1's mixture: below it the floor carries more
#: weight than the GCE term, which is S3's hard-prune reading.
LAMBDA_HANDOVER = 0.5


@dataclass
class DesignConfig:
    """The round-2 declared configuration, over the round-1 world config.

    Every field is declared before any run; nothing is tuned after seeing
    outcomes. `base` carries the world, the λ grid and the branch schedule
    round 1 declared, unchanged, so the two rounds' tables are comparable.
    """

    base: StudyConfig = None
    seeds: tuple = tuple(range(30))
    theta_warmup_steps: int = THETA_WARMUP_STEPS
    theta_arm_steps: int = THETA_ARM_STEPS
    theta_lr: float = THETA_LR
    q: float = DESIGN_Q

    def __post_init__(self):
        if self.base is None:
            self.base = StudyConfig()

    def as_dict(self) -> dict:
        return {
            "base": self.base.as_dict(),
            "seeds": list(self.seeds),
            "thetaWarmupSteps": self.theta_warmup_steps,
            "thetaArmSteps": self.theta_arm_steps,
            "thetaLr": self.theta_lr,
            "q": self.q,
            "featureMeans": [list(m) for m in FEATURE_MEANS],
            "lambdaHandover": LAMBDA_HANDOVER,
        }


def generate_features(config: DesignConfig, truth: torch.Tensor,
                      seed: int) -> torch.Tensor:
    """The declared 2-D features: class-conditional Gaussians per pixel.

    x_n ~ N(mu_{k*(n)}, I) with the means on the unit circle — drawn from this
    run's own torch.Generator so the world's random stream stays exactly round
    1's. Deterministic in (truth, seed).
    """
    k = config.base.n_classes
    if truth.ndim != 1 or int(truth.max()) >= k:
        raise ValueError(f"truth must carry labels in [0, {k}); got {truth!r}")
    means = torch.tensor(FEATURE_MEANS[:k], dtype=torch.float64)
    gen = torch.Generator().manual_seed(int(seed))
    return means[truth] + torch.randn(truth.shape[0], 2, generator=gen,
                                      dtype=torch.float64)


def generate_design_world(config: DesignConfig, scenario: str,
                          seed: int) -> dict:
    """Round 1's world, unchanged, plus the declared features.

    The shared tensors (truth, region, labels, mask, P, Y, agree) are
    `selection_study.generate_world`'s own output for the same (scenario,
    seed) — bitwise, which `design_world_shares_round_one_world` asserts — and
    `features` is the one addition.
    """
    world = generate_world(config.base, scenario, seed)
    world["features"] = generate_features(config, world["truth"], seed)
    return world


class LinearSegmenter(torch.nn.Module):
    """The declared segmenter: linear softmax over the 2-D features."""

    def __init__(self, config: DesignConfig):
        super().__init__()
        k = config.base.n_classes
        self.weight = torch.nn.Parameter(torch.zeros(k, 2, dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.zeros(k, dtype=torch.float64))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(features @ self.weight.T + self.bias, dim=-1)


def arm_weight(arm: str, gate: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """The arm's weight w on the GCE term: 1{P > C̃} (S1), g·Λ (S2), Λ (S3)."""
    if arm == "S1":
        return gate
    if arm == "S2":
        return gate * lam
    if arm == "S3":
        return lam
    raise ValueError(
        f"unknown arm {arm!r}; the round declares exactly {list(ARMS)}")


def arm_integrand(arm: str, P: torch.Tensor, Y: torch.Tensor, M: torch.Tensor,
                  Lambda: torch.Tensor, cutoffs: torch.Tensor,
                  q: float) -> torch.Tensor:
    """The arm's masked integrand of Eq. (1); shape [R, ...].

    `integrand = M · ( w · GCE(P[k_r]) + (1 − w) · floor(C̃_{k_r}) )` with the
    arm's weight w (`arm_weight`). The gate 1{P[k_r] > C̃_{k_r}} and Λ are
    constants in a segmentation step — φ never trains by Eq. 1 — so the floor
    contributes no gradient and P's gradient at every entry is scaled by w.
    S3's integrand is Eq. (1) exactly (`s3_arm_is_eq1`).
    """
    if arm not in ARMS:
        raise ValueError(
            f"unknown arm {arm!r}; the round declares exactly {list(ARMS)}")
    labels = annotated_labels(Y)
    n_classes = P.shape[-1]
    if P.shape[:-1] != labels.shape[1:]:
        raise ValueError(
            f"P must have shape [*S, {n_classes}] with labels [R, *S]; got "
            f"P {tuple(P.shape)}, labels {tuple(labels.shape)}")
    if Lambda.shape != labels.shape + (n_classes,):
        raise ValueError(
            f"Lambda must have shape {tuple(labels.shape + (n_classes,))}; "
            f"got {tuple(Lambda.shape)}")
    if M.shape != labels.shape:
        raise ValueError(f"M must have shape {tuple(labels.shape)}; got {tuple(M.shape)}")
    P_exp = P.unsqueeze(0).expand(labels.shape[0], *P.shape)
    p_annot = gather_at_annotated(P_exp, labels)
    lam = gather_at_annotated(Lambda, labels).detach()
    c_annot = cutoffs.to(P.dtype)[labels]
    gate = (p_annot > c_annot).to(P.dtype).detach()
    w = arm_weight(arm, gate, lam)
    gce = (1 - p_annot ** float(q)) / float(q)
    floor = (1 - c_annot ** float(q)) / float(q)
    return M.to(P.dtype) * (w * gce + (1 - w) * floor)


def crowd_gce(P: torch.Tensor, Y: torch.Tensor, M: torch.Tensor,
              q: float) -> torch.Tensor:
    """The warm-up loss: plain crowd GCE over all annotated entries — no
    truncation, no reliability (r05 A3: truncation is enabled only after the
    warm-up). Identical for every arm."""
    labels = annotated_labels(Y)
    P_exp = P.unsqueeze(0).expand(labels.shape[0], *P.shape)
    p_annot = gather_at_annotated(P_exp, labels)
    gce = (1 - p_annot ** float(q)) / float(q)
    return (M.to(P.dtype) * gce).mean()


def train_segmenter(world: dict, thresholds: dict, config: DesignConfig,
                    trained_lambda: torch.Tensor, arm: str) -> dict:
    """Train the linear segmenter through ONE arm's loss, in two phases.

    Phase 1 (warm-up, identical for every arm): `theta_warmup_steps` steps of
    plain crowd GCE. Phase 2: `theta_arm_steps` steps of the arm's integrand
    mean, the gate and Λ held constant inside each step (Λ is the shared
    C3-trained reliability; the gate is recomputed from the current P at every
    step, as the arm's mechanism reads). Returns the trained P, the two loss
    trajectories and the step counts.
    """
    if arm not in ARMS:
        raise ValueError(
            f"unknown arm {arm!r}; the round declares exactly {list(ARMS)}")
    theta = LinearSegmenter(config)
    optimizer = torch.optim.Adam(theta.parameters(), lr=config.theta_lr)
    features = world["features"]
    Y, M = world["Y"], world["mask"].to(torch.float64)
    cutoffs = thresholds["cutoffs"]
    warmup_losses: list = []
    arm_losses: list = []
    for _ in range(int(config.theta_warmup_steps)):
        optimizer.zero_grad()
        loss = crowd_gce(theta(features), Y, M, config.q)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(
                "the warm-up produced a non-finite loss; the configuration is "
                "defective and training fails closed")
        loss.backward()
        optimizer.step()
        warmup_losses.append(float(loss.detach()))
    P = None
    with torch.no_grad():
        P = theta(features)
    for _ in range(int(config.theta_arm_steps)):
        optimizer.zero_grad()
        P = theta(features)
        loss = arm_integrand(arm, P, Y, M, trained_lambda,
                             cutoffs, config.q).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(
                f"arm {arm} produced a non-finite loss; the configuration is "
                "defective and training fails closed")
        loss.backward()
        optimizer.step()
        arm_losses.append(float(loss.detach()))
    with torch.no_grad():
        P_final = theta(features)
    return {"P": P_final.detach().clone(), "warmupLosses": warmup_losses,
            "armLosses": arm_losses}


def consensus_metrics(P: torch.Tensor, truth: torch.Tensor) -> dict:
    """The primary metric: consensus quality against the synthetic truth.

    mIoU: mean over classes of the intersection-over-union of argmax P against
    the consensus labels; a class absent from BOTH sides contributes no value
    (the mean runs over the classes present in either). Dice: per class
    2|A∩B|/(|A|+|B|), scored 1.0 when both sides are empty (the two agree the
    class is absent), averaged over all classes. P error: the cross-entropy
    −mean log P[n, k*] — the true probability field is the point mass on the
    consensus, declared.
    """
    if P.ndim != 2 or P.shape[0] != truth.shape[0]:
        raise ValueError(
            f"P must be [N, K] over N pixels; got {tuple(P.shape)} for truth "
            f"{tuple(truth.shape)}")
    if not bool(torch.isfinite(P).all()):
        raise ValueError("P must be finite")
    k = P.shape[-1]
    pred = P.argmax(dim=-1)
    ious = []
    dices = []
    for cat in range(k):
        a = pred == cat
        b = truth == cat
        union = int((a | b).sum())
        inter = int((a & b).sum())
        if union:
            ious.append(inter / union)
        sa, sb = int(a.sum()), int(b.sum())
        dices.append(1.0 if sa + sb == 0 else 2 * inter / (sa + sb))
    miou = sum(ious) / len(ious) if ious else float("nan")
    ce = -torch.log(P[torch.arange(P.shape[0]), truth].clamp_min(1e-300)).mean()
    return {"mIoU": float(miou), "Dice": float(sum(dices) / len(dices)),
            "pErrorCE": float(ce), "perClassIoU": ious}


def survivor_mask(world: dict, trained_lambda: torch.Tensor, cutoffs: torch.Tensor,
                  arm: str) -> torch.Tensor:
    """The arm's own surviving annotations, at the evaluated entries.

    S1/S2 survive where the gate 1{P[n, k_r] > C̃_{k_r}} is 1 — their hard
    prune is the gate. S3 survives where Λ ≥ 0.5 — the point where Eq. 1's
    mixture hands the weight from the GCE term to the floor; below it the
    reliability has pruned the annotation. Raises ValueError on an unknown
    arm.
    """
    if arm not in ARMS:
        raise ValueError(
            f"unknown arm {arm!r}; the round declares exactly {list(ARMS)}")
    mask = world["mask"]
    rows, cols = mask.nonzero(as_tuple=True)
    labels_at = world["labels"][rows, cols]
    if arm == "S3":
        lam_at = trained_lambda[rows, cols, labels_at]
        return lam_at >= LAMBDA_HANDOVER
    p_exp = world["P"].unsqueeze(0).expand(mask.shape[0], *world["P"].shape)
    p_annot = p_exp[rows, cols, labels_at]
    return p_annot > cutoffs.to(p_annot.dtype)[labels_at]


def arm_pruning(world: dict, trained_lambda: torch.Tensor, cutoffs: torch.Tensor,
                arm: str) -> dict:
    """The arm's pruning precision/recall against the known-wrong annotations.

    Each arm's own hard prune (`survivor_mask`'s complement), scored against
    the annotations the generator knows are wrong (y ≠ k*). Precision and
    recall are micro-averaged over the categories; NaN where undefined
    (nothing pruned, or no wrong annotation) — reported, never invented.
    """
    mask = world["mask"]
    rows, cols = mask.nonzero(as_tuple=True)
    labels_at = world["labels"][rows, cols]
    truth_at = world["truth"][cols]
    wrong_at = world["labels"][rows, cols] != truth_at
    survivors = survivor_mask(world, trained_lambda, cutoffs, arm)
    pruned = ~survivors
    pruned_total = int(pruned.sum())
    pruned_wrong = int((pruned & wrong_at).sum())
    overall_wrong = int(wrong_at.sum())
    return {
        "precision": pruned_wrong / pruned_total if pruned_total else float("nan"),
        "recall": pruned_wrong / overall_wrong if overall_wrong else float("nan"),
        "prunedTotal": pruned_total,
        "annotatedTotal": int(mask.sum()),
    }


def surviving_lambda_mean(world: dict, trained_lambda: torch.Tensor,
                          cutoffs: torch.Tensor, arm: str) -> list:
    """The mean Λ over the arm's surviving annotations, per category.

    The statistic that will later decide whether the weight needs
    normalization: S2's survivors are the gate's, S3's are Λ ≥ 0.5. A category
    with no surviving annotation is NaN — reported, never invented.
    """
    mask = world["mask"]
    rows, cols = mask.nonzero(as_tuple=True)
    labels_at = world["labels"][rows, cols]
    survivors = survivor_mask(world, trained_lambda, cutoffs, arm)
    lam_at = trained_lambda[rows, cols, labels_at].to(torch.float64)
    out = []
    for cat in range(world["P"].shape[-1]):
        pick = survivors & (labels_at == cat)
        out.append(float(lam_at[pick].mean()) if bool(pick.any()) else float("nan"))
    return out


def run_design_block(config: DesignConfig, scenario: str, seed: int) -> dict:
    """One (scenario, seed) block: the shared estimation, then the three arms.

    The shared half — world, warm-up estimation (Eq. 3 → Eq. 4), the
    C3-trained reliability (Ec. 7 with the segmentation held fixed, the
    round-1 protocol), the Λ metrics and the case-A sensor — is computed once
    and is identical for every arm. Each arm then trains its own segmenter
    (declared warm-up + arm phase) and reports the primary consensus metrics,
    its own pruning precision/recall, and (S2, S3) the mean Λ over its
    survivors per category.
    """
    world = generate_design_world(config, scenario, seed)
    thresholds = estimate_thresholds(world, config.base)
    trained = train_candidate(world, thresholds, config.base, "C3")
    lam = trained["lambda"]
    shared = {
        "thresholds": {"rhoHat": thresholds["rho_hat"].tolist(),
                       "cutoffs": thresholds["cutoffs"].tolist(),
                       "discarded": thresholds["discarded"].tolist()},
        "sensor": floor_sensor(world, thresholds),
        "lambdaMetrics": {
            "recoveryError": recovery_error(lam, world),
            "gateCorrelation": gate_correlation(lam, world,
                                                thresholds["cutoffs"]),
            "collapse": collapse_diagnosis(lam, world),
        },
        "finalObjective": float(trained["finalObjective"]),
    }
    arms = {}
    for arm in ARMS:
        result = train_segmenter(world, thresholds, config, lam, arm)
        metrics = consensus_metrics(result["P"], world["truth"])
        entry = {
            "primary": metrics,
            "pruning": arm_pruning(world, lam, thresholds["cutoffs"], arm),
            "warmupLossHead": result["warmupLosses"][:3],
            "warmupLossLast": result["warmupLosses"][-1],
            "armLossLast": result["armLosses"][-1],
        }
        if arm in ("S2", "S3"):
            entry["survivingLambda"] = surviving_lambda_mean(
                world, lam, thresholds["cutoffs"], arm)
        arms[arm] = entry
    return {"scenario": scenario, "seed": int(seed), "shared": shared,
            "arms": arms}
