"""Randomized configuration sweep shared by the audit and the remedy checks.

One fixed size for every claim: a property surviving SWEEP_SIZE independent
configurations belongs to the formulation, not to a seed.

SWEEP_BASE offsets the block by the run's seed so separate runs audit disjoint
configurations. Without it, agreement between runs would confirm determinism
and nothing else.

`configuration` builds one instance of the proposal's objects: a model
confidence field P, one-hot annotations (Y, M), and a per-category noise
estimate rho_hat -- the inputs of Eq. (4)'s calibration. Every configuration is
constructed so the 1/K floor of Eq. (4) binds in every category: each category's
confidence sample is a mixture with a heavy mass strictly below 1/K and the rest
strictly above it, and rho_hat_k is drawn strictly below F_k(1/K). That makes
the clip observable -- quantile_k < 1/K while the cutoff is floored at 1/K --
without hard-coding the outcome: the magnitudes, the category count, the
annotator count, the sample sizes and q all vary per index.

Torch-only backend: one `torch.Generator` per index, seeded with
`SWEEP_BASE + index`, on CPU so sweeps agree across devices.
"""

from __future__ import annotations

import torch

SWEEP_SIZE = 200
SWEEP_BASE = 7 * 10000


def configuration(index: int) -> dict:
    """One reproducible configuration of the proposal's objects.

    Not `conftest.py`'s shared `rng` fixture: this is a plain function, never
    a collected pytest item, so it cannot receive a fixture at all -- and it
    needs a generator seeded per `index`, which the fixture's one fixed
    `SEED` does not provide anyway. The two stay separate on purpose.

    Returns {"index", "K", "R", "N", "q", "P", "Y", "M", "rho_hat",
    "floor_binds", "low_fractions"}. `floor_binds` names the categories whose
    rho_hat-quantile sits strictly below 1/K by construction (all of them here);
    `low_fractions[k]` is the exact F_k(1/K) the construction produces.
    """
    rng = torch.Generator().manual_seed(SWEEP_BASE + index)
    k_classes = int(torch.randint(2, 6, (1,), generator=rng).item())
    r_annotators = int(torch.randint(1, 4, (1,), generator=rng).item())
    q = 0.5 + 0.5 * float(torch.rand(1, generator=rng).item())
    n_per_category = [int(torch.randint(20, 60, (1,), generator=rng).item())
                      for _ in range(k_classes)]
    n_extra = int(torch.randint(10, 30, (1,), generator=rng).item())
    n_pixels = sum(n_per_category) + n_extra

    P = torch.zeros(n_pixels, k_classes, dtype=torch.float64)
    # Y is one-hot at every (r, n); entries with M = 0 carry an arbitrary label
    # and are never read as annotations.
    labels_grid = torch.randint(0, k_classes, (r_annotators, n_pixels),
                                generator=rng)
    M = torch.zeros(r_annotators, n_pixels, dtype=torch.float64)
    rho_hat = torch.zeros(k_classes, dtype=torch.float64)
    low_fractions = []

    start = 0
    for k in range(k_classes):
        n_k = n_per_category[k]
        rows = torch.arange(start, start + n_k)
        start += n_k

        # The confidence sample of category k: a heavy low cluster strictly
        # below 1/K and the rest strictly above it. F_k(1/K) is then exactly
        # the low fraction, and the rho_hat_k-quantile of any level below it
        # lands strictly below 1/K -- the floor binds.
        p_low = 0.55 + 0.4 * float(torch.rand(1, generator=rng).item())
        n_low = max(2, min(n_k - 1, int(round(p_low * n_k))))
        low_span = (0.3 + 0.6 * float(torch.rand(1, generator=rng).item())) / k_classes
        lows = torch.rand(n_low, generator=rng, dtype=torch.float64) * low_span
        highs = (1.0 / k_classes + (1.0 - 1.0 / k_classes)
                 * (0.05 + 0.95 * torch.rand(n_k - n_low, generator=rng,
                                             dtype=torch.float64)))
        values = torch.cat([lows, highs])
        values = values[torch.randperm(n_k, generator=rng)]

        P[rows, k] = values
        for j in range(k_classes):
            if j != k:
                P[rows, j] = (1.0 - values) / (k_classes - 1)

        # Exactly one annotation per pixel of the k block, on a random
        # annotator: the pixel contributes P[pixel, k] to category k's F_k
        # sample and nothing to any other category's.
        who = torch.randint(0, r_annotators, (n_k,), generator=rng)
        for i, r in enumerate(who.tolist()):
            n = int(rows[i])
            labels_grid[r, n] = k
            M[r, n] = 1.0

        low_fractions.append(n_low / n_k)
        # Strictly below F_k(1/K) = n_low / n_k, so the discarded fraction at
        # the floored cutoff exceeds rho_hat strictly in every configuration.
        u = 0.02 + 0.48 * float(torch.rand(1, generator=rng).item())
        rho_hat[k] = u * (n_low / n_k)

    # Unannotated pixels (M = 0): they must never enter any F_k sample.
    extra = torch.arange(start, start + n_extra)
    P[extra] = torch.softmax(
        torch.randn(n_extra, k_classes, generator=rng, dtype=torch.float64), dim=-1)

    Y = torch.nn.functional.one_hot(labels_grid, k_classes).to(torch.float64)
    return {
        "index": index,
        "K": k_classes,
        "R": r_annotators,
        "N": n_pixels,
        "q": q,
        "P": P,
        "Y": Y,
        "M": M,
        "rho_hat": rho_hat,
        "floor_binds": list(range(k_classes)),
        "low_fractions": low_fractions,
    }


def sweep(size: int = SWEEP_SIZE):
    for index in range(size):
        yield configuration(index)
