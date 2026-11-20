"""Unified Test benchmark & paired statistical comparison across all 5 methods.

Evaluates the three trained segmentation models (``CT-GCE``, ``Annot-Harmony``,
``CrowdSeg``) loaded from saved ``.pth`` checkpoints alongside the two
non-parametric multi-annotator baselines (``MV``, ``Spatial-STAPLE``) on the
same Test split and annotator consensus target, then computes paired t-tests
and paired Wilcoxon signed-rank tests with Holm-Bonferroni correction against
``--reference`` (default: ``CT-GCE``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from crowdsegmenter.config import ExperimentConfig
from crowdsegmenter.data.loader import CrowdSegmenterDataLoader
from crowdsegmenter.models.annot_harmony import AnnotHarmony
from crowdsegmenter.models.baselines import MajorityVoting, SpatialSTAPLE
from crowdsegmenter.models.crowdseg import CrowdSeg
from crowdsegmenter.models.ctgce import CTGCEModel
from crowdsegmenter.models.supervised import SupervisedResUNet
from crowdsegmenter.utils.ablation import (
    compare_per_sample_metrics,
    load_per_sample_npz,
    save_ablation_report,
)
from crowdsegmenter.utils.metrics import (
    compute_loader_panel_nlc_metrics,
    extract_prepared_dense_features,
)
from crowdsegmenter.utils.reproducibility import set_seed
from scripts._cli_utils import _enable_realtime_logs, evaluate_on_test, serialize_tensors

DEFAULT_DATASET_CONFIGS: Dict[str, Dict[str, str]] = {
    "oxfordiiitpet": {
        "CT-GCE": "configs/training/ctgce/oxfordiiitpet.yaml",
        "Annot-Harmony": "configs/training/tgce_ssps/oxfordiiitpet.yaml",
        "CrowdSeg": "configs/training/crowdseg/oxfordiiitpet.yaml",
        "MV": "configs/training/MV/oxfordiiitpet.yaml",
        "Spatial-STAPLE": "configs/training/spatial_staple/oxfordiiitpet.yaml",
        "Supervised-GT": "configs/training/supervised/oxfordiiitpet.yaml",
    },
    "riga": {
        "CT-GCE": "configs/training/ctgce/riga.yaml",
        "Annot-Harmony": "configs/training/tgce_ssps/riga.yaml",
        "CrowdSeg": "configs/training/crowdseg/riga.yaml",
        "MV": "configs/training/MV/riga.yaml",
        "Spatial-STAPLE": "configs/training/spatial_staple/riga.yaml",
    },
    "naip_wellpad": {
        "CT-GCE": "configs/training/ctgce/naip_wellpad.yaml",
        "Annot-Harmony": "configs/training/tgce_ssps/naip_wellpad.yaml",
        "CrowdSeg": "configs/training/crowdseg/naip_wellpad.yaml",
        "MV": "configs/training/MV/naip_wellpad.yaml",
        "Spatial-STAPLE": "configs/training/spatial_staple/naip_wellpad.yaml",
        "Supervised-GT": "configs/training/supervised/naip_wellpad.yaml",
    },
}


@dataclass(frozen=True)
class _MethodSpec:
    name: str
    slug: str
    builder: Callable[[ExperimentConfig, Optional[float]], nn.Module]
    weights_key: Optional[str] = None


def _build_neural_model(
    cls: type[nn.Module],
    cfg: ExperimentConfig,
) -> nn.Module:
    # Avoid downloading ImageNet weights before overwriting parameters from checkpoint
    infer_model_cfg = cfg.model.model_copy(update={"pretrained": False})
    return cls(infer_model_cfg)


def _load_checkpoint(
    model: nn.Module,
    weights_path: str | Path,
    device: torch.device,
) -> nn.Module:
    ckpt_path = Path(weights_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(payload, dict):
        if "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        elif "state_dict" in payload:
            state_dict = payload["state_dict"]
        else:
            state_dict = payload
    else:
        state_dict = payload

    model.load_state_dict(state_dict)
    model.eval()
    return model


def _build_method_specs(include_supervised: bool = False) -> tuple[_MethodSpec, ...]:
    specs = [
        _MethodSpec(
            name="CT-GCE",
            slug="ct_gce",
            weights_key="CT-GCE",
            builder=lambda cfg, _: _build_neural_model(CTGCEModel, cfg),
        ),
        _MethodSpec(
            name="Annot-Harmony",
            slug="annot_harmony",
            weights_key="Annot-Harmony",
            builder=lambda cfg, _: _build_neural_model(AnnotHarmony, cfg),
        ),
        _MethodSpec(
            name="CrowdSeg",
            slug="crowdseg",
            weights_key="CrowdSeg",
            builder=lambda cfg, _: _build_neural_model(CrowdSeg, cfg),
        ),
        _MethodSpec(
            name="MV",
            slug="mv",
            builder=lambda cfg, _: MajorityVoting(
                num_annotators=cfg.model.num_annotators,
                num_classes=cfg.model.num_classes,
                ignored_value=cfg.data.ignored_value,
            ),
        ),
        _MethodSpec(
            name="Spatial-STAPLE",
            slug="spatial_staple",
            builder=lambda cfg, beta: SpatialSTAPLE(
                num_annotators=cfg.model.num_annotators,
                num_classes=cfg.model.num_classes,
                ignored_value=cfg.data.ignored_value,
                beta=beta if beta is not None else cfg.model.staple_beta,
                max_iter=cfg.model.staple_max_iter,
                tol=cfg.model.staple_tol,
            ),
        ),
    ]
    if include_supervised:
        specs.append(
            _MethodSpec(
                name="Supervised-GT",
                slug="supervised",
                weights_key="Supervised-GT",
                builder=lambda cfg, _: _build_neural_model(SupervisedResUNet, cfg),
            )
        )
    return tuple(specs)


def _print_benchmark_table(
    rows: List[Dict[str, Any]],
    protocol: str,
    alpha: float,
) -> None:
    if not rows:
        return
    title = f" BENCHMARK & STATISTICAL TESTS ({protocol.upper()}) "
    print(f"\n{title:=^124}")
    print(
        f"{'Method':<16} | {'Metric':<11} | {'Mean ± Std':<17} | "
        f"{'Delta':<9} | {'t-test p':<11} | {'t-test p_adj':<12} | "
        f"{'Wilcoxon p_adj':<14} | {'Sig (t/W @ ' + str(alpha) + ')'}"
    )
    print("-" * 124)
    for r in rows:
        mean_std = f"{r['mean']:.4f} ± {r['std']:.4f}"
        sig_t = r["ttest_p_adj"] < alpha
        sig_w = r["wilcoxon_p_adj"] < alpha
        print(
            f"{r['method']:<16} | {r['metric']:<11} | {mean_std:<17} | "
            f"{r['delta_vs_ref']:+8.4f} | {r['ttest_p']:<11.4g} | {r['ttest_p_adj']:<12.4g} | "
            f"{r['wilcoxon_p_adj']:<14.4g} | t={sig_t} / W={sig_w}"
        )
    print("=" * 124)


def _print_nlc_table(
    nlc_rows: List[Dict[str, Any]],
    panel_nlc: Optional[Dict[str, Any]] = None,
) -> None:
    if not nlc_rows:
        return
    cls_keys = [k for k in nlc_rows[0] if k.startswith("class_")]
    header_cls = " | ".join(f"{k:<10}" for k in cls_keys)
    width = 62 + len(cls_keys) * 13
    print(f"\n{' BENCHMARK ZERO-GT NLC CONSISTENCY ':=^{width}}")
    print(
        f"{'Method':<22} | {'Macro Avg':<10} | {'Delta':<9} | "
        + (header_cls if cls_keys else "")
    )
    print("-" * width)
    for r in nlc_rows:
        cls_vals = " | ".join(f"{r[k]:<10.4f}" for k in cls_keys)
        print(
            f"{r['method']:<22} | {r['macro_avg']:<10.4f} | {r['delta_vs_ref']:+8.4f} | "
            + cls_vals
        )
    if panel_nlc is not None:
        p_cls = " | ".join(
            f"{panel_nlc['per_class'][k]['panel_mean']:<10.4f}"
            for k in cls_keys
            if k in panel_nlc.get("per_class", {})
        )
        print("-" * width)
        print(
            f"{'Annotator Panel (mean)':<22} | {panel_nlc['macro_panel_mean']:<10.4f} | {'—':<9} | "
            + p_cls
        )
    print("=" * width)


def run_benchmark_evaluation(
    ctgce_weights: str | Path,
    annotharmony_weights: str | Path,
    crowdseg_weights: str | Path,
    supervised_weights: Optional[str | Path] = None,
    dataset: Optional[str] = "oxfordiiitpet",
    config_overrides: Optional[Mapping[str, str | Path]] = None,
    data_dir: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    reference: str = "CT-GCE",
    alpha: float = 0.05,
    beta: Optional[float] = None,
    eval_nlc: Optional[bool] = None,
    nlc_pretrained: bool = True,
) -> Dict[str, Any]:
    """Evaluate all methods (5 crowd + optional Supervised GT oracle) on Test."""
    _enable_realtime_logs()

    method_specs = _build_method_specs(include_supervised=supervised_weights is not None)
    defaults = DEFAULT_DATASET_CONFIGS.get(dataset or "", {})
    resolved_configs: Dict[str, Path] = {}
    for spec in method_specs:
        override = (config_overrides or {}).get(spec.name)
        cfg_val = override if override is not None else defaults.get(spec.name)
        if cfg_val is None:
            raise ValueError(
                f"No YAML config resolved for {spec.name!r} (dataset={dataset!r})."
            )
        resolved_configs[spec.name] = Path(cfg_val)

    weights_map: Dict[str, Path] = {
        "CT-GCE": Path(ctgce_weights),
        "Annot-Harmony": Path(annotharmony_weights),
        "CrowdSeg": Path(crowdseg_weights),
    }
    if supervised_weights is not None:
        weights_map["Supervised-GT"] = Path(supervised_weights)
    for name, wpath in weights_map.items():
        if not wpath.is_file():
            raise FileNotFoundError(f"Missing checkpoint for {name}: {wpath}")

    configs: Dict[str, ExperimentConfig] = {}
    for name, cfg_path in resolved_configs.items():
        cfg = ExperimentConfig.from_yaml(cfg_path)
        if data_dir is not None:
            cfg.data.data_dir = str(data_dir)
        cfg.experiment.save_results = True
        configs[name] = cfg

    ref_cfg = configs[reference]
    device = torch.device(ref_cfg.training.device if torch.cuda.is_available() else "cpu")
    set_seed(ref_cfg.training.seed)

    out_root = Path(
        output_dir
        if output_dir is not None
        else f"results/benchmark/{dataset or 'custom'}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    use_nlc = ref_cfg.training.eval_nlc if eval_nlc is None else eval_nlc

    print(f"\n{'':=<72}")
    print(f"  5-Method Benchmark Evaluation ({dataset or 'custom'})")
    print(f"  Device    : {device}")
    print(f"  Reference : {reference}")
    print(f"  Output    : {out_root}")
    print(f"  GT eval   : {ref_cfg.data.load_ground_truth}")
    print(f"  Zero-GT   : {use_nlc}")
    print(f"{'':=<72}\n")

    # Use all-annotator mode so probabilistic consensus targets are identical across all 5 methods
    data_manager = CrowdSegmenterDataLoader(ref_cfg.data, mode="CT-GCE")
    _, _, test_loader = data_manager.get_split_loaders()

    nlc_features = (
        extract_prepared_dense_features(
            loader=test_loader,
            device=device,
            pretrained=nlc_pretrained,
        )
        if use_nlc
        else None
    )

    method_dirs: Dict[str, Path] = {}
    method_summaries: Dict[str, Any] = {}

    for spec in method_specs:
        cfg = configs[spec.name]
        model_dir = out_root / spec.slug
        model_dir.mkdir(parents=True, exist_ok=True)

        model = spec.builder(cfg, beta).to(device)
        if spec.weights_key is not None:
            _load_checkpoint(model, weights_map[spec.weights_key], device=device)
        else:
            model.eval()

        test_summary = evaluate_on_test(
            model=model,
            test_loader=test_loader,
            cfg=cfg,
            device=device,
            output_path=model_dir,
            label=spec.name,
            prefix="test",
            eval_nlc=use_nlc,
            nlc_features=nlc_features,
            nlc_pretrained=nlc_pretrained,
        )
        method_dirs[spec.name] = model_dir
        method_summaries[spec.name] = test_summary

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    protocols: List[Literal["probabilistic", "gt"]] = ["probabilistic"]
    if ref_cfg.data.load_ground_truth:
        protocols.append("gt")

    comparisons: Dict[str, List[Dict[str, Any]]] = {}
    for proto in protocols:
        arrays = {
            name: load_per_sample_npz(m_dir / f"test_{proto}_per_sample.npz")
            for name, m_dir in method_dirs.items()
        }
        rows = compare_per_sample_metrics(arrays, reference=reference, alpha=alpha)
        _print_benchmark_table(rows, protocol=proto, alpha=alpha)
        save_ablation_report(rows, output_dir=out_root, stem=f"benchmark_{proto}")
        comparisons[proto] = rows

    nlc_rows: List[Dict[str, Any]] = []
    panel_nlc: Optional[Dict[str, Any]] = None
    if use_nlc and nlc_features is not None:
        panel_nlc = compute_loader_panel_nlc_metrics(
            features=nlc_features,
            loader=test_loader,
            num_classes=ref_cfg.model.num_classes,
            num_annotators=ref_cfg.model.num_annotators,
            ignored_value=ref_cfg.data.ignored_value,
            prepare=False,
        )
        ref_macro = float(method_summaries[reference]["nlc"]["macro_avg"])
        for spec in method_specs:
            nlc_info = method_summaries[spec.name]["nlc"]
            row: Dict[str, Any] = {
                "method": spec.name,
                "reference": reference,
                "macro_avg": float(nlc_info["macro_avg"]),
                "delta_vs_ref": float(ref_macro - nlc_info["macro_avg"]),
            }
            for cls_k, cls_val in nlc_info["per_class"].items():
                row[cls_k] = float(cls_val)
            nlc_rows.append(row)
        _print_nlc_table(nlc_rows, panel_nlc=panel_nlc)
        save_ablation_report(nlc_rows, output_dir=out_root, stem="benchmark_nlc")

    summary_payload: Dict[str, Any] = {
        "dataset": dataset,
        "reference": reference,
        "alpha": alpha,
        "eval_nlc": use_nlc,
        "weights": {k: str(v) for k, v in weights_map.items()},
        "configs": {k: str(v) for k, v in resolved_configs.items()},
        "methods": serialize_tensors(method_summaries),
        "comparisons": serialize_tensors(comparisons),
    }
    if use_nlc:
        summary_payload["nlc_comparison"] = serialize_tensors(nlc_rows)
        summary_payload["panel_nlc"] = serialize_tensors(panel_nlc)
    summary_path = out_root / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=4)

    print(f"\nBenchmark complete. Summary & paired tests saved in: {out_root}")
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified 5-method Test evaluation & paired t-test/Wilcoxon benchmark"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=sorted(DEFAULT_DATASET_CONFIGS.keys()),
        default="oxfordiiitpet",
        help="Dataset preset selecting default training YAML configs.",
    )
    parser.add_argument(
        "--ctgce_weights",
        type=str,
        required=True,
        help="Path to trained CT-GCE checkpoint (.pth).",
    )
    parser.add_argument(
        "--tgce_ssps_weights",
        "--annotharmony_weights",
        dest="tgce_ssps_weights",
        type=str,
        required=True,
        help="Path to trained TGCE-SSPS (Annot-Harmony) checkpoint (.pth).",
    )
    parser.add_argument(
        "--crowdseg_weights",
        type=str,
        required=True,
        help="Path to trained CrowdSeg checkpoint (.pth).",
    )
    parser.add_argument(
        "--supervised_weights",
        type=str,
        default=None,
        help="Optional path to trained Supervised-GT oracle checkpoint (.pth).",
    )
    parser.add_argument("--ctgce_config", type=str, default=None, help="Optional CT-GCE YAML override.")
    parser.add_argument(
        "--tgce_ssps_config",
        "--annotharmony_config",
        dest="tgce_ssps_config",
        type=str,
        default=None,
        help="Optional TGCE-SSPS (Annot-Harmony) YAML override.",
    )
    parser.add_argument("--crowdseg_config", type=str, default=None, help="Optional CrowdSeg YAML override.")
    parser.add_argument("--mv_config", type=str, default=None, help="Optional MV YAML override.")
    parser.add_argument("--staple_config", type=str, default=None, help="Optional Spatial-STAPLE YAML override.")
    parser.add_argument("--supervised_config", type=str, default=None, help="Optional Supervised-GT YAML override.")
    parser.add_argument("--data", type=str, default=None, help="Optional override for data_dir.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional override for output_dir.")
    parser.add_argument(
        "--reference",
        type=str,
        default="CT-GCE",
        choices=[
            "CT-GCE",
            "Annot-Harmony",
            "CrowdSeg",
            "MV",
            "Spatial-STAPLE",
            "Supervised-GT",
        ],
        help="Reference method for paired t-test and Wilcoxon tests.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level after Holm-Bonferroni correction.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Optional Potts MRF beta override for Spatial-STAPLE.",
    )
    parser.add_argument(
        "--eval-nlc",
        "--eval_nlc",
        dest="eval_nlc",
        action="store_true",
        default=None,
        help="Compute zero-GT NLC consistency across all 5 methods and annotator panel.",
    )
    args = parser.parse_args()

    overrides: Dict[str, str] = {}
    for key, val in (
        ("CT-GCE", args.ctgce_config),
        ("Annot-Harmony", args.tgce_ssps_config),
        ("CrowdSeg", args.crowdseg_config),
        ("MV", args.mv_config),
        ("Spatial-STAPLE", args.staple_config),
        ("Supervised-GT", args.supervised_config),
    ):
        if val is not None:
            overrides[key] = val

    run_benchmark_evaluation(
        ctgce_weights=args.ctgce_weights,
        annotharmony_weights=args.tgce_ssps_weights,
        crowdseg_weights=args.crowdseg_weights,
        supervised_weights=args.supervised_weights,
        dataset=args.dataset,
        config_overrides=overrides,
        data_dir=args.data,
        output_dir=args.output_dir,
        reference=args.reference,
        alpha=args.alpha,
        beta=args.beta,
        eval_nlc=args.eval_nlc,
    )


if __name__ == "__main__":
    main()
