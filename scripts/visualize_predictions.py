"""Save visual inference panels (input, ordinal masks, GT, lambdas, CMs) per sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from crowdsegmenter.config import ExperimentConfig
from crowdsegmenter.data.loader import CrowdSegmenterDataLoader
from crowdsegmenter.utils.reproducibility import set_seed
from crowdsegmenter.visualization import (
    annotator_masks_to_ordinal,
    extract_sample_artifact,
    gt_to_ordinal,
    save_sample_arrays,
    save_sample_figure,
)
from scripts._cli_utils import (
    DEFAULT_DATASET_CONFIGS,
    MethodSpec,
    _enable_realtime_logs,
    load_checkpoint,
    resolve_method_selection,
    resolve_reference_method,
)

_SPLIT_INDEX = {"train": 0, "valid": 1, "val": 1, "test": 2}


def _has_foreground(
    sample_tuple: Tuple[torch.Tensor, ...],
    num_annotators: int,
    num_classes: int,
    ignored_value: float,
) -> bool:
    if len(sample_tuple) > 3:
        return bool((gt_to_ordinal(sample_tuple[3]) > 0).any())
    ann_ord, _ = annotator_masks_to_ordinal(
        sample_tuple[1],
        num_annotators=num_annotators,
        num_classes=num_classes,
        ignored_value=ignored_value,
    )
    return bool((ann_ord > 0).any())


def _select_sample_indices(
    dataset: Any,
    num_samples: int,
    sample_indices: Optional[Sequence[int]],
    only_positive: bool,
    num_annotators: int,
    num_classes: int,
    ignored_value: float,
) -> List[int]:
    n_total = len(dataset)
    if n_total == 0:
        raise ValueError("Selected split dataset is empty.")

    if sample_indices is not None:
        chosen: List[int] = []
        for idx in sample_indices:
            if idx < 0 or idx >= n_total:
                raise IndexError(f"Sample index {idx} out of range [0, {n_total - 1}].")
            if idx not in chosen:
                chosen.append(int(idx))
        return chosen

    target_count = min(max(int(num_samples), 1), n_total)
    if not only_positive:
        step = max(1, n_total // target_count)
        return list(range(0, n_total, step))[:target_count]

    # Probe evenly across the split to collect positive samples quickly without decoding all tiles
    candidates: List[int] = []
    probe_step = max(1, n_total // (target_count * 4))
    probe_order = list(range(0, n_total, probe_step))
    seen = set(probe_order)
    probe_order.extend(i for i in range(n_total) if i not in seen)

    fallback: List[int] = []
    for idx in probe_order:
        if _has_foreground(dataset[idx], num_annotators, num_classes, ignored_value):
            candidates.append(idx)
            if len(candidates) >= target_count:
                break
        elif len(fallback) < target_count:
            fallback.append(idx)

    if len(candidates) < target_count:
        candidates.extend(i for i in fallback if i not in candidates)
    return candidates[:target_count]


def run_visual_predictions(
    ctgce_weights: Optional[str | Path] = None,
    annotharmony_weights: Optional[str | Path] = None,
    crowdseg_weights: Optional[str | Path] = None,
    supervised_weights: Optional[str | Path] = None,
    include_mv: bool = True,
    include_staple: bool = True,
    methods: Optional[Sequence[str]] = None,
    dataset: Optional[str] = "oxfordiiitpet",
    config_overrides: Optional[Mapping[str, str | Path]] = None,
    data_dir: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    split: str = "test",
    num_samples: int = 6,
    sample_indices: Optional[Sequence[int]] = None,
    only_positive: bool = True,
    save_npz: bool = False,
    beta: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run inference on selected samples and save ordinal + lambda/CM visual figures."""
    _enable_realtime_logs()

    method_specs, weights_map, resolved_configs = resolve_method_selection(
        ctgce_weights=ctgce_weights,
        annotharmony_weights=annotharmony_weights,
        crowdseg_weights=crowdseg_weights,
        supervised_weights=supervised_weights,
        include_mv=include_mv,
        include_staple=include_staple,
        methods=methods,
        dataset=dataset,
        config_overrides=config_overrides,
    )
    ref_name = resolve_reference_method(method_specs)

    configs: Dict[str, ExperimentConfig] = {}
    for name, cfg_path in resolved_configs.items():
        cfg = ExperimentConfig.from_yaml(cfg_path)
        if data_dir is not None:
            cfg.data.data_dir = str(data_dir)
        configs[name] = cfg

    ref_cfg = configs[ref_name]
    device = torch.device(ref_cfg.training.device if torch.cuda.is_available() else "cpu")
    set_seed(seed if seed is not None else ref_cfg.training.seed)

    split_key = split.strip().lower()
    if split_key not in _SPLIT_INDEX:
        raise ValueError(f"Unsupported split {split!r}; expected train, valid, or test.")

    out_root = Path(
        output_dir
        if output_dir is not None
        else f"results/visualizations/{dataset or 'custom'}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    data_manager = CrowdSegmenterDataLoader(ref_cfg.data, mode="CT-GCE")
    loaders = data_manager.get_split_loaders()
    target_dataset = loaders[_SPLIT_INDEX[split_key]].dataset

    indices = _select_sample_indices(
        dataset=target_dataset,
        num_samples=num_samples,
        sample_indices=sample_indices,
        only_positive=only_positive,
        num_annotators=ref_cfg.model.num_annotators,
        num_classes=ref_cfg.model.num_classes,
        ignored_value=ref_cfg.data.ignored_value,
    )

    model_entries: List[Tuple[MethodSpec, ExperimentConfig, nn.Module]] = []
    for spec in method_specs:
        cfg = configs[spec.name]
        model = spec.builder(cfg, beta).to(device)
        if spec.weights_key is not None:
            load_checkpoint(model, weights_map[spec.weights_key], device=device)
        else:
            model.eval()
        model_entries.append((spec, cfg, model))

    print(f"\n{'':=<72}")
    print(f"  Visual Prediction Export ({len(model_entries)} methods, {dataset or 'custom'})")
    print(f"  Split     : {split} (indices={indices})")
    print(f"  Methods   : {', '.join(s.name for s in method_specs)}")
    print(f"  Device    : {device}")
    print(f"  Output    : {out_root}")
    print(f"{'':=<72}\n")

    class_names = [f"Class {k}" for k in range(ref_cfg.model.num_classes)]
    saved_entries: List[Dict[str, Any]] = []

    for idx in indices:
        sample_tuple = target_dataset[idx]
        file_name = (
            target_dataset.file_names[idx]
            if hasattr(target_dataset, "file_names")
            else f"sample_{idx}.png"
        )
        stem = Path(file_name).stem
        artifact = extract_sample_artifact(
            sample_index=idx,
            file_name=file_name,
            sample_tuple=sample_tuple,
            model_entries=model_entries,
            ref_cfg=ref_cfg,
            device=device,
        )
        fig_path = save_sample_figure(
            artifact,
            output_path=out_root / f"sample_{idx:04d}_{stem}.png",
            class_names=class_names,
        )
        npz_path = (
            save_sample_arrays(artifact, out_root / f"sample_{idx:04d}_{stem}_arrays.npz")
            if save_npz
            else None
        )
        entry = {
            "sample_index": idx,
            "file_name": file_name,
            "figure_path": str(fig_path),
            "arrays_path": str(npz_path) if npz_path is not None else None,
            "sample_dice": {p.name: p.sample_dice for p in artifact.predictions},
        }
        saved_entries.append(entry)
        dice_summary = ", ".join(
            f"{k}={v:.3f}" for k, v in entry["sample_dice"].items() if v is not None
        )
        print(f"  Saved {fig_path.name} ({dice_summary})", flush=True)

    summary_payload: Dict[str, Any] = {
        "dataset": dataset,
        "split": split,
        "methods": [s.name for s in method_specs],
        "weights": {k: str(v) for k, v in weights_map.items()},
        "samples": saved_entries,
    }
    with open(out_root / "visual_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=4)

    print(f"\nVisual export complete. Figures saved in: {out_root}")
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save ordinal predictions, annotator/GT masks, lambdas, and CMs per sample"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=sorted(DEFAULT_DATASET_CONFIGS.keys()),
        default="oxfordiiitpet",
        help="Dataset preset selecting default training YAML configs.",
    )
    parser.add_argument("--ctgce_weights", type=str, default=None, help="Optional CT-GCE checkpoint (.pth).")
    parser.add_argument(
        "--tgce_ssps_weights",
        "--annotharmony_weights",
        dest="tgce_ssps_weights",
        type=str,
        default=None,
        help="Optional TGCE-SSPS (Annot-Harmony) checkpoint (.pth).",
    )
    parser.add_argument("--crowdseg_weights", type=str, default=None, help="Optional CrowdSeg checkpoint (.pth).")
    parser.add_argument("--supervised_weights", type=str, default=None, help="Optional Supervised-GT checkpoint (.pth).")
    parser.add_argument(
        "--no-mv",
        "--no_mv",
        dest="include_mv",
        action="store_false",
        default=True,
        help="Exclude Majority Voting when --methods is not specified.",
    )
    parser.add_argument(
        "--no-staple",
        "--no_staple",
        dest="include_staple",
        action="store_false",
        default=True,
        help="Exclude Spatial-STAPLE when --methods is not specified.",
    )
    parser.add_argument("--methods", nargs="+", default=None, help="Explicit subset of methods to visualize.")
    parser.add_argument("--ctgce_config", type=str, default=None, help="Optional CT-GCE YAML override.")
    parser.add_argument(
        "--tgce_ssps_config",
        "--annotharmony_config",
        dest="tgce_ssps_config",
        type=str,
        default=None,
        help="Optional TGCE-SSPS YAML override.",
    )
    parser.add_argument("--crowdseg_config", type=str, default=None, help="Optional CrowdSeg YAML override.")
    parser.add_argument("--mv_config", type=str, default=None, help="Optional MV YAML override.")
    parser.add_argument("--staple_config", type=str, default=None, help="Optional Spatial-STAPLE YAML override.")
    parser.add_argument("--supervised_config", type=str, default=None, help="Optional Supervised-GT YAML override.")
    parser.add_argument("--data", type=str, default=None, help="Optional override for data_dir.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional override for output_dir.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=6, help="Number of samples to export.")
    parser.add_argument(
        "--sample_indices",
        nargs="+",
        type=int,
        default=None,
        help="Explicit dataset indices to render (overrides --num_samples).",
    )
    parser.add_argument(
        "--all-samples",
        "--all_samples",
        dest="only_positive",
        action="store_false",
        default=True,
        help="Pick evenly spaced samples without filtering for positive foreground.",
    )
    parser.add_argument(
        "--save-npz",
        "--save_npz",
        dest="save_npz",
        action="store_true",
        default=False,
        help="Also save raw ordinal masks, lambdas, and CMs in per-sample .npz files.",
    )
    parser.add_argument("--beta", type=float, default=None, help="Optional Potts beta for Spatial-STAPLE.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
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

    run_visual_predictions(
        ctgce_weights=args.ctgce_weights,
        annotharmony_weights=args.tgce_ssps_weights,
        crowdseg_weights=args.crowdseg_weights,
        supervised_weights=args.supervised_weights,
        include_mv=args.include_mv,
        include_staple=args.include_staple,
        methods=args.methods,
        dataset=args.dataset,
        config_overrides=overrides,
        data_dir=args.data,
        output_dir=args.output_dir,
        split=args.split,
        num_samples=args.num_samples,
        sample_indices=args.sample_indices,
        only_positive=args.only_positive,
        save_npz=args.save_npz,
        beta=args.beta,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
