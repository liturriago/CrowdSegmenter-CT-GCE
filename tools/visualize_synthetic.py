#!/usr/bin/env python3
"""Render one PNG per synthetic family used by the CT-GCE test suite.

This script implements no equation of the formulation: it only draws samples
that the tests and the audit already construct. It is an inspection aid, not a
source of truth -- `tests/` remains the source of truth.

Run it with the target repository's own interpreter:

    cd implementations/CrowdSegmenter-CT-GCE
    .venv/bin/python tools/visualize_synthetic.py

Outputs, under `results/synthetic/`:
    01-algebraic-fixture.png   the hand-built Eq. (1)-(2) fixture
    02-synthetic-crowd.png     a sample of the Dawid-Skene crowd model
    03-audit-sweep.png         the 200-configuration Eq. (4) sweep

`matplotlib` is a rendering dependency only; it is not a dependency of the
formulation. If it is missing, install it into the target venv:
    uv pip install --python .venv/bin/python "matplotlib>=3.8"
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from CrowdSegmenter_CT_GCE.dawid_skene import majority_posterior  # noqa: E402
from CrowdSegmenter_CT_GCE.module import ctgce_loss  # noqa: E402
from CrowdSegmenter_CT_GCE.reliability import (  # noqa: E402
    annotated_labels, gather_at_annotated)
from CrowdSegmenter_CT_GCE.thresholds import (  # noqa: E402
    calibrate_cutoffs, category_confidences)
from sweep import sweep  # noqa: E402

OUT = ROOT / "results" / "synthetic"
OUT.mkdir(parents=True, exist_ok=True)

CATEGORICAL = ListedColormap(["#4C72B0", "#DD8452", "#55A868"])
MASKED = "#D9D9D9"
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.grid": False,
    "font.size": 9,
}
CLASS_LEGEND = [
    Patch(facecolor="#4C72B0", label="class 0"),
    Patch(facecolor="#DD8452", label="class 1"),
    Patch(facecolor="#55A868", label="class 2"),
    Patch(facecolor=MASKED, label="not annotated (M=0)"),
]


def _save(fig, name: str) -> Path:
    path = OUT / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(path.relative_to(ROOT))
    return path


def _class_map(ax, labels, title, mask=None, annotate=True):
    """A categorical label map, with masked entries drawn grey."""
    values = labels.detach().cpu().to(torch.int64).numpy()
    shown = values.astype(float)
    if mask is not None:
        shown = shown.copy()
        shown[mask.detach().cpu().numpy() == 0] = float("nan")
    cmap = CATEGORICAL.copy()
    cmap.set_bad(MASKED)
    ax.imshow(shown, cmap=cmap, vmin=-0.5, vmax=2.5, interpolation="nearest")
    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if mask is not None and int(mask[i, j]) == 0:
                    ax.text(j, i, "M=0", ha="center", va="center", fontsize=7,
                            color="#666666")
                else:
                    ax.text(j, i, str(int(values[i, j])), ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _heat(ax, values, title, cmap="Blues", vmin=0.0, vmax=1.0, fmt="{:.2f}",
          annotate=True):
    array = values.detach().cpu().to(torch.float64).numpy()
    ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    if annotate:
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                ax.text(j, i, fmt.format(array[i, j]), ha="center", va="center",
                        fontsize=7,
                        color="white" if array[i, j] > (vmin + vmax) / 2
                        else "#222222")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Family 1 -- the hand-built algebraic fixture (tests/test_invariants.py)
# ---------------------------------------------------------------------------
def family_one() -> None:
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
    labels = annotated_labels(Y)
    lambda_annot = gather_at_annotated(Lambda, labels)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(10, 6.6))
        _heat(axes[0, 0], P, "P: class probabilities [pixel, k]")
        axes[0, 0].set_xticks(range(3))
        axes[0, 0].set_xticklabels(["k=0", "k=1", "k=2"], fontsize=7)
        axes[0, 0].set_yticks(range(3))
        axes[0, 0].set_yticklabels(["pixel 0", "pixel 1", "pixel 2"], fontsize=7)
        _class_map(axes[0, 1], labels[0].reshape(1, 3), "r=0 annotated class k_r")
        _class_map(axes[0, 2], labels[1].reshape(1, 3), "r=1 annotated class k_r")
        _heat(axes[1, 0], lambda_annot[0].reshape(1, 3), "Lambda at k_r, r=0", cmap="viridis")
        _heat(axes[1, 1], lambda_annot[1].reshape(1, 3), "Lambda at k_r, r=1", cmap="viridis")
        _class_map(axes[1, 2], labels[1].reshape(1, 3), "M_r over r=1's labels",
                   mask=M[1].reshape(1, 3))
        fig.suptitle(
            "Family 1 - algebraic fixture: R=2, 3 pixels, K=3; Eq. (1)-(2)\n"
            f"CT-GCE loss = {float(ctgce_loss(P, Y, M, Lambda, C, q)):.6f} (fixture Lambda)   "
            f"q = {q}   C = {C.tolist()}",
            fontsize=10)
        fig.legend(handles=CLASS_LEGEND[:3], loc="lower center", ncol=3,
                   frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        _save(fig, "01-algebraic-fixture.png")


# ---------------------------------------------------------------------------
# Family 2 -- a sample of the Dawid-Skene crowd model (tests/test_synthetic.py)
# ---------------------------------------------------------------------------
def _crowd_sample(seed: int = 7, n_classes: int = 3, n_annotators: int = 4,
                  side: int = 16):
    rng = torch.Generator().manual_seed(seed)
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

    n = side * side
    truth = torch.multinomial(prior, n, replacement=True, generator=rng)
    labels = torch.empty(n_annotators, n, dtype=torch.int64)
    for r in range(n_annotators):
        flip_prob = 1 - confusions[r].diagonal()[truth]
        is_flip = torch.rand(n, generator=rng, dtype=torch.float64) < flip_prob
        alt = torch.randint(0, n_classes - 1, (n,), generator=rng)
        alt = alt + (alt >= truth).to(torch.int64)
        labels[r] = torch.where(is_flip, alt, truth)

    majority = majority_posterior(labels, torch.ones_like(labels, dtype=torch.bool),
                                  n_classes).argmax(dim=1)
    return (truth.reshape(side, side), labels.reshape(n_annotators, side, side),
            majority.reshape(side, side), flips)


def _empirical_confusion(truth, labels, n_classes=3):
    flat_truth = truth.reshape(-1).to(torch.int64)
    flat_labels = labels.reshape(-1).to(torch.int64)
    counts = torch.bincount(flat_truth * n_classes + flat_labels,
                            minlength=n_classes * n_classes).reshape(
                                n_classes, n_classes).to(torch.float64)
    return counts / counts.sum(dim=1, keepdim=True).clamp_min(1)


def family_two() -> None:
    truth, labels, majority, flips = _crowd_sample()
    accuracy = [(labels[r] == truth).to(torch.float64).mean().item()
                for r in range(labels.shape[0])]
    empirical = _empirical_confusion(truth, labels[0])

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 3, figsize=(11, 10))
        _class_map(axes[0, 0], truth, "latent consensus k*", annotate=False)
        for index, ax in enumerate([axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]):
            _class_map(ax, labels[index], f"annotator r={index}", annotate=False)
        _class_map(axes[1, 2], majority, "majority vote", annotate=False)

        ax = axes[2, 0]
        ax.bar(range(labels.shape[0]), accuracy, color="#4C72B0")
        ax.axhline(1 / 3, color="#C44E52", linestyle="--", linewidth=1,
                   label="chance 1/K")
        ax.set_ylim(0, 1)
        ax.set_xticks(range(labels.shape[0]))
        ax.set_xlabel("annotator r")
        ax.set_ylabel("agreement with k*")
        ax.set_title("per-annotator accuracy", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(axis="y", color="#E5E5E5")

        ax = axes[2, 1]
        for r in range(labels.shape[0]):
            ax.plot(range(3), flips[r].tolist(), marker="o", label=f"r={r}")
        ax.set_xticks(range(3))
        ax.set_xlabel("category k")
        ax.set_ylabel("flip rate")
        ax.set_ylim(0, 0.35)
        ax.set_title("true per-annotator flip rates", fontsize=9)
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax.grid(axis="y", color="#E5E5E5")

        _heat(axes[2, 2], empirical, "r=0 empirical confusion T_0[k,k']")
        axes[2, 2].set_xticks(range(3))
        axes[2, 2].set_xticklabels(["k'=0", "k'=1", "k'=2"], fontsize=7)
        axes[2, 2].set_yticks(range(3))
        axes[2, 2].set_yticklabels(["k=0", "k=1", "k=2"], fontsize=7)

        fig.suptitle(
            "Family 2 - synthetic crowd: R=4, K=3, 16x16 pixels; "
            "A1 conditional independence\n"
            "truth ~ prior [0.5, 0.3, 0.2]; each annotator flips with its own "
            "per-category rate and picks uniformly among the other classes",
            fontsize=10)
        fig.legend(handles=CLASS_LEGEND, loc="lower center", ncol=4,
                   frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        _save(fig, "02-synthetic-crowd.png")


# ---------------------------------------------------------------------------
# Family 3 -- the 200-configuration Eq. (4) sweep (tests/sweep.py)
# ---------------------------------------------------------------------------
def family_three() -> None:
    rho, discarded, clipped, gaps = [], [], [], []
    first = None
    for config in sweep():
        calibration = calibrate_cutoffs(config["P"], config["Y"], config["M"],
                                        config["rho_hat"])
        for k in range(config["K"]):
            r = float(config["rho_hat"][k])
            d = float(calibration.discarded[k])
            rho.append(r)
            discarded.append(d)
            clipped.append(bool(calibration.clipped[k]))
            gaps.append(d - r)
        if first is None:
            first = (config, calibration)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))

        ax = axes[0, 0]
        ax.scatter(rho, discarded, s=14, alpha=0.55, color="#C44E52",
                   edgecolors="none", label="category, one point")
        limit = max(max(rho), max(discarded)) * 1.05
        ax.plot([0, limit], [0, limit], color="#777777", linestyle="--",
                linewidth=1, label="y = rho_hat (exact)")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_xlabel("rho_hat_k")
        ax.set_ylabel("discarded fraction F_k(C_k)")
        ax.set_title("every point sits above the diagonal", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(color="#E5E5E5")

        ax = axes[0, 1]
        ax.hist(gaps, bins=30, color="#55A868", edgecolor="white")
        ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
        ax.set_xlabel("F_k(C_k) - rho_hat_k")
        ax.set_ylabel("category-cases (all 200 configurations)")
        ax.set_title(f"gap: min {min(gaps):.3f}, median "
                     f"{sorted(gaps)[len(gaps) // 2]:.3f}", fontsize=9)
        ax.grid(axis="y", color="#E5E5E5")

        config, calibration = first
        ax = axes[1, 0]
        k_classes = config["K"]
        x = list(range(k_classes))
        ax.bar([i - 0.2 for i in x], config["rho_hat"].tolist(), width=0.4,
               color="#4C72B0", label="rho_hat")
        ax.bar([i + 0.2 for i in x], calibration.discarded.tolist(), width=0.4,
               color="#C44E52", label="discarded F_k(C_k)")
        ax.set_xticks(x)
        ax.set_xlabel("category k")
        ax.set_ylabel("fraction")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"config 0: K={k_classes}, clipped="
                     f"{calibration.clipped.tolist()}", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(axis="y", color="#E5E5E5")

        ax = axes[1, 1]
        sample = category_confidences(config["P"], config["Y"], config["M"])[0]
        ordered = torch.sort(sample).values
        ecdf = torch.arange(1, ordered.numel() + 1, dtype=torch.float64) / ordered.numel()
        ax.step(ordered.tolist(), ecdf.tolist(), where="post", color="#4C72B0",
                label="F_0")
        ax.axvline(float(calibration.quantile[0]), color="#DD8452",
                   linestyle="--", linewidth=1.2, label="quantile")
        ax.axvline(float(calibration.cutoffs[0]), color="#C44E52",
                   linestyle=":", linewidth=1.4, label="C_0 = 1/K")
        ax.axhline(float(config["rho_hat"][0]), color="#55A868",
                   linestyle="-.", linewidth=1, label="rho_hat_0")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("confidence P[pixel, 0] on pixels labeled 0")
        ax.set_ylabel("F_0(t)")
        ax.set_title("config 0, category 0: the clip raises C_0", fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax.grid(color="#E5E5E5")

        fig.suptitle(
            "Family 3 - audit sweep: 200 configurations, 1/K floor binds by "
            "construction; Eq. (4)\n"
            "the honest discarded fraction is F_k(C_k), not rho_hat_k",
            fontsize=10)
        fig.tight_layout()
        _save(fig, "03-audit-sweep.png")


if __name__ == "__main__":
    family_one()
    family_two()
    family_three()
