"""Statistical hypothesis testing and ablation comparison utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Apply Holm-Bonferroni step-down correction to a sequence of p-values."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    adjusted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, float(p_values[idx]) * (m - rank))
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return [float(x) for x in adjusted]


def load_per_sample_npz(path: str | Path) -> Dict[str, np.ndarray]:
    """Load per-sample metric arrays from a ``*_per_sample.npz`` file."""
    npz_path = Path(path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Per-sample metrics file not found: {npz_path}")
    with np.load(npz_path) as data:
        return {k: np.asarray(data[k], dtype=np.float64) for k in data.files}


def _safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    if diff.size == 0 or np.allclose(diff, 0.0):
        return 1.0
    _, pval = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(pval) if np.isfinite(pval) else 1.0


def _safe_paired_ttest(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    if diff.size < 2 or np.allclose(diff, diff[0]):
        return 1.0 if np.allclose(diff, 0.0) else 0.0
    _, pval = ttest_rel(a, b)
    return float(pval) if np.isfinite(pval) else 1.0


def compare_per_sample_metrics(
    method_arrays: Dict[str, Dict[str, np.ndarray]],
    reference: str,
    metrics: Sequence[str] = ("dice", "jaccard", "sensitivity", "specificity"),
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """Compare each method against ``reference`` using paired Wilcoxon & t-tests.

    Applies Holm-Bonferroni correction per metric across all non-reference methods.
    """
    if reference not in method_arrays:
        raise ValueError(
            f"Reference method {reference!r} not found in {sorted(method_arrays.keys())}"
        )

    ref_data = method_arrays[reference]
    competitors = [m for m in method_arrays if m != reference]
    rows: List[Dict[str, Any]] = []

    # Summary row for reference first
    for metric in metrics:
        if metric not in ref_data:
            continue
        ref_vals = ref_data[metric]
        rows.append(
            {
                "method": reference,
                "reference": reference,
                "metric": metric,
                "n": int(ref_vals.size),
                "mean": float(np.mean(ref_vals)),
                "std": float(np.std(ref_vals, ddof=1)) if ref_vals.size > 1 else 0.0,
                "median": float(np.median(ref_vals)),
                "delta_vs_ref": 0.0,
                "wilcoxon_p": 1.0,
                "wilcoxon_p_adj": 1.0,
                "ttest_p": 1.0,
                "ttest_p_adj": 1.0,
                "significant": False,
            }
        )

        raw_w: List[float] = []
        raw_t: List[float] = []
        pending: List[Dict[str, Any]] = []

        for comp in competitors:
            comp_vals = method_arrays[comp][metric]
            if comp_vals.shape != ref_vals.shape:
                raise ValueError(
                    f"Sample count mismatch for {comp}/{metric}: "
                    f"{comp_vals.shape} vs {ref_vals.shape}"
                )
            p_w = _safe_wilcoxon(ref_vals, comp_vals)
            p_t = _safe_paired_ttest(ref_vals, comp_vals)
            raw_w.append(p_w)
            raw_t.append(p_t)
            pending.append(
                {
                    "method": comp,
                    "reference": reference,
                    "metric": metric,
                    "n": int(comp_vals.size),
                    "mean": float(np.mean(comp_vals)),
                    "std": float(np.std(comp_vals, ddof=1)) if comp_vals.size > 1 else 0.0,
                    "median": float(np.median(comp_vals)),
                    "delta_vs_ref": float(np.mean(ref_vals - comp_vals)),
                    "wilcoxon_p": p_w,
                    "ttest_p": p_t,
                }
            )

        adj_w = holm_bonferroni(raw_w)
        adj_t = holm_bonferroni(raw_t)
        for item, pw_adj, pt_adj in zip(pending, adj_w, adj_t):
            item["wilcoxon_p_adj"] = pw_adj
            item["ttest_p_adj"] = pt_adj
            item["significant"] = pw_adj < alpha
            rows.append(item)

    return rows


def save_ablation_report(
    rows: List[Dict[str, Any]],
    output_dir: str | Path,
    stem: str = "ablation_comparison",
) -> tuple[Path, Path]:
    """Persist ablation comparison rows to JSON and CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{stem}.json"
    csv_path = out / f"{stem}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=4)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    return json_path, csv_path
