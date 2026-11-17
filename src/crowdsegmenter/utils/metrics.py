"""Segmentation evaluation metrics and zero-GT Neighborhood Latent Consistency (NLC)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from CrowdSegmenter_CT_GCE.metric import (
    consistency as nlc_consistency,
    panel_lower_quantile,
    panel_mean,
    panel_median,
    prepare_features,
    prototype as nlc_prototype,
)


class MetricTracker:
    """Computes segmentation metrics under probabilistic consensus and ground-truth protocols."""

    def __init__(self, config: Any) -> None:
        self.num_classes = config.num_classes
        self.num_annotators = config.num_annotators
        self.ignored_value = config.ignored_value
        self.threshold = config.threshold
        self.probabilistic_thresholds = config.probabilistic_thresholds
        self.smooth = config.smooth

    def compute_probability_mask(self, masks: torch.Tensor) -> torch.Tensor:
        """Build soft consensus mask ``[B, C, H, W]`` from ``[B, A*C, H, W]`` (layout ``k*A + r``)."""
        expected = self.num_annotators * self.num_classes
        if masks.shape[1] != expected:
            raise ValueError(
                f"Expected masks with {expected} channels "
                f"(num_annotators={self.num_annotators} x num_classes={self.num_classes}), "
                f"but received shape {tuple(masks.shape)}."
            )

        b = masks.shape[0]
        masks_reshaped = masks.reshape(
            b, self.num_classes, self.num_annotators, *masks.shape[2:]
        ).transpose(1, 2)  # [B, A, C, H, W]

        valid_mask = (masks_reshaped != self.ignored_value).float()
        valid_count = valid_mask.sum(dim=1)
        masks_sum = (masks_reshaped * valid_mask).sum(dim=1)

        return torch.where(
            valid_count > 0,
            masks_sum / valid_count,
            torch.zeros_like(masks_sum),
        )

    def calculate_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute Dice, Jaccard (IoU), sensitivity, and specificity at a single threshold."""
        threshold = threshold if threshold is not None else self.threshold

        y_pred = y_pred.contiguous().float()
        y_true = y_true.contiguous().float()

        mask = (y_true != self.ignored_value).float()
        y_true_bin = (y_true > threshold).float()
        y_pred_bin = (y_pred > threshold).float()

        tp = (y_true_bin * y_pred_bin * mask).sum(dim=(2, 3))
        fp = ((1 - y_true_bin) * y_pred_bin * mask).sum(dim=(2, 3))
        fn = (y_true_bin * (1 - y_pred_bin) * mask).sum(dim=(2, 3))
        tn = ((1 - y_true_bin) * (1 - y_pred_bin) * mask).sum(dim=(2, 3))

        dice = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        jaccard = (tp + self.smooth) / (tp + fp + fn + self.smooth)
        sensitivity = (tp + self.smooth) / (tp + fn + self.smooth)
        specificity = (tn + self.smooth) / (tn + fp + self.smooth)

        def _nan_to_zero(t: torch.Tensor) -> torch.Tensor:
            return torch.where(torch.isnan(t), torch.zeros_like(t), t)

        dice, jaccard, sensitivity, specificity = map(
            _nan_to_zero, (dice, jaccard, sensitivity, specificity)
        )

        return {
            "avg": {
                "dice": dice.mean().item(),
                "jaccard": jaccard.mean().item(),
                "sensitivity": sensitivity.mean().item(),
                "specificity": specificity.mean().item(),
            },
            "per_class": {
                "dice": dice.mean(dim=0),
                "jaccard": jaccard.mean(dim=0),
                "sensitivity": sensitivity.mean(dim=0),
                "specificity": specificity.mean(dim=0),
            },
            "per_sample": {
                "dice": dice.mean(dim=1),
                "jaccard": jaccard.mean(dim=1),
                "sensitivity": sensitivity.mean(dim=1),
                "specificity": specificity.mean(dim=1),
            },
        }

    def compute_probabilistic_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> Dict[str, Any]:
        """Average segmentation metrics across ``self.probabilistic_thresholds``."""
        num_thresholds = len(self.probabilistic_thresholds)
        num_classes = y_pred.shape[1]
        num_samples = y_pred.shape[0]

        avg_sums = {"dice": 0.0, "jaccard": 0.0, "sensitivity": 0.0, "specificity": 0.0}
        per_class_sums = {
            key: torch.zeros(num_classes, device=y_pred.device) for key in avg_sums
        }
        per_sample_sums = {
            key: torch.zeros(num_samples, device=y_pred.device) for key in avg_sums
        }

        for threshold in self.probabilistic_thresholds:
            metrics = self.calculate_metrics(y_pred, y_true, threshold)
            for key in avg_sums:
                avg_sums[key] += metrics["avg"][key]
                per_class_sums[key] += metrics["per_class"][key]
                per_sample_sums[key] += metrics["per_sample"][key]

        return {
            "avg": {k: v / num_thresholds for k, v in avg_sums.items()},
            "per_class": {k: v / num_thresholds for k, v in per_class_sums.items()},
            "per_sample": {k: v / num_thresholds for k, v in per_sample_sums.items()},
        }

    def save_per_sample_results(
        self,
        per_sample_metrics: Dict[str, torch.Tensor],
        output_dir: Union[str, Path],
        protocol: str,
        prefix: str = "test",
    ) -> None:
        """Save per-sample metrics to an ``.npz`` archive."""
        if protocol not in ("probabilistic", "gt"):
            raise ValueError(f"Invalid protocol: {protocol}. Must be 'probabilistic' or 'gt'.")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / f"{prefix}_{protocol}_per_sample.npz"

        data_to_save = {
            key: tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else np.array(tensor)
            for key, tensor in per_sample_metrics.items()
        }
        np.savez(filepath, **data_to_save)

    def evaluation(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: Union[str, torch.device],
        save_results: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        prefix: str = "test",
    ) -> Dict[str, Any]:
        """Run inference and compute consensus probabilistic metrics."""
        model.eval()
        all_seg_preds: List[torch.Tensor] = []
        all_prob_masks: List[torch.Tensor] = []
        is_baseline = getattr(model, "is_baseline", False)

        with torch.no_grad():
            for batch in loader:
                masks = batch[1].to(device)
                anns_ids = batch[2].to(device)
                if is_baseline:
                    output = model(masks, anns_ids)
                else:
                    images = batch[0].to(device)
                    output = model(images, anns_ids)
                seg_pred = output[0] if isinstance(output, tuple) else output
                all_seg_preds.append(seg_pred.cpu())
                all_prob_masks.append(self.compute_probability_mask(masks).cpu())

        flat_preds = torch.cat(all_seg_preds, dim=0)
        flat_masks = torch.cat(all_prob_masks, dim=0)
        full_metrics = self.compute_probabilistic_metrics(flat_preds, flat_masks)

        if save_results:
            if output_dir is None:
                raise ValueError("output_dir must be provided when save_results is True.")
            self.save_per_sample_results(
                per_sample_metrics=full_metrics["per_sample"],
                output_dir=output_dir,
                protocol="probabilistic",
                prefix=prefix,
            )

        return full_metrics

    def compute_gt_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> Dict[str, Any]:
        """Compute ground-truth metrics at ``self.threshold``."""
        return self.calculate_metrics(y_pred, y_true, self.threshold)

    def evaluation_gt(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: Union[str, torch.device],
        save_results: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        prefix: str = "test",
    ) -> Dict[str, Any]:
        """Run inference and compute ground-truth metrics over ``loader``."""
        model.eval()
        all_seg_preds: List[torch.Tensor] = []
        all_gt_masks: List[torch.Tensor] = []
        is_baseline = getattr(model, "is_baseline", False)

        with torch.no_grad():
            for batch in loader:
                if len(batch) < 4:
                    raise ValueError(
                        f"Batch does not contain ground truth (len={len(batch)} < 4). "
                        "Set load_ground_truth=True in DataConfig."
                    )
                masks = batch[1].to(device)
                anns_ids = batch[2].to(device)
                ground_truth = batch[3].to(device)
                if is_baseline:
                    output = model(masks, anns_ids)
                else:
                    images = batch[0].to(device)
                    output = model(images, anns_ids)
                seg_pred = output[0] if isinstance(output, tuple) else output
                all_seg_preds.append(seg_pred.cpu())
                all_gt_masks.append(ground_truth.cpu())

        flat_preds = torch.cat(all_seg_preds, dim=0)
        flat_gt = torch.cat(all_gt_masks, dim=0)
        full_metrics = self.compute_gt_metrics(flat_preds, flat_gt)

        if save_results:
            if output_dir is None:
                raise ValueError("output_dir must be provided when save_results is True.")
            self.save_per_sample_results(
                per_sample_metrics=full_metrics["per_sample"],
                output_dir=output_dir,
                protocol="gt",
                prefix=prefix,
            )

        return full_metrics

    def evaluation_nlc(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: Union[str, torch.device],
        features: Optional[Sequence[np.ndarray]] = None,
        nlc_pretrained: bool = True,
    ) -> Dict[str, Any]:
        """Run inference and compute zero-GT NLC consistency across foreground classes."""
        model.eval()
        all_seg_preds: List[torch.Tensor] = []
        is_baseline = getattr(model, "is_baseline", False)

        with torch.no_grad():
            for batch in loader:
                masks = batch[1].to(device)
                anns_ids = batch[2].to(device)
                if is_baseline:
                    output = model(masks, anns_ids)
                else:
                    images = batch[0].to(device)
                    output = model(images, anns_ids)
                seg_pred = output[0] if isinstance(output, tuple) else output
                all_seg_preds.append(seg_pred.cpu())

        flat_preds = torch.cat(all_seg_preds, dim=0)
        prep_feats = (
            list(features)
            if features is not None
            else extract_prepared_dense_features(
                loader=loader,
                device=device,
                pretrained=nlc_pretrained,
            )
        )
        return compute_prediction_nlc_metrics(
            features=prep_feats,
            seg_preds=flat_preds,
            num_classes=self.num_classes,
            threshold=self.threshold,
            prepare=False,
        )

    @staticmethod
    def print_summary(
        prefix: str,
        metrics: Dict[str, Any],
        class_names: List[str],
    ) -> None:
        avg = metrics["avg"]
        print(
            f"\n[{prefix}] Dice: {avg['dice']:.4f} | "
            f"Jaccard: {avg['jaccard']:.4f} | "
            f"Sens.: {avg['sensitivity']:.4f} | "
            f"Spec.: {avg['specificity']:.4f}"
        )
        print(
            f"  > {'Class':<15} | {'Dice':<8} | {'Jaccard':<8} | "
            f"{'Sens.':<8} | {'Spec.':<8}"
        )
        print(f"  {'-' * 57}")

        pc = metrics["per_class"]
        for i, name in enumerate(class_names):
            print(
                f"    {name:<15} | {pc['dice'][i]:.4f}   | "
                f"{pc['jaccard'][i]:.4f}   | "
                f"{pc['sensitivity'][i]:.4f}   | "
                f"{pc['specificity'][i]:.4f}"
            )

    @staticmethod
    def print_full_report(
        prefix: str,
        metrics: Dict[str, Any],
        class_names: List[str],
    ) -> None:
        print(f"\n{' REPORT: ' + prefix + ' ':=^85}")
        header = (
            f"{'Class':<20} | {'Dice':<8} | {'Jaccard':<8} | "
            f"{'Sens.':<8} | {'Spec.':<8}"
        )
        print(header)
        print("-" * len(header))

        pc = metrics["per_class"]
        for i, name in enumerate(class_names):
            print(
                f"{name:<20} | {pc['dice'][i]:<8.4f} | "
                f"{pc['jaccard'][i]:<8.4f} | "
                f"{pc['sensitivity'][i]:<8.4f} | "
                f"{pc['specificity'][i]:<8.4f}"
            )

        print("-" * len(header))
        avg = metrics["avg"]
        print(
            f"{'Macro Avg':<20} | {avg['dice']:<8.4f} | "
            f"{avg['jaccard']:<8.4f} | "
            f"{avg['sensitivity']:<8.4f} | "
            f"{avg['specificity']:<8.4f}"
        )
        print("=" * len(header))


def compute_nlc_panel_metrics(
    features: Sequence[np.ndarray],
    annotator_masks: Sequence[Sequence[np.ndarray]],
    prepare: bool = True,
) -> Dict[str, Any]:
    """Compute r10 zero-GT Neighborhood Latent Consistency (Eqs. 8-9) for an annotator panel.

    Args:
        features: Sequence of ``N`` dense feature maps ``[C, H, W]`` (``N >= 2``).
        annotator_masks: For each annotator ``a in 0..R-1``, a sequence of ``N``
            binary/soft foreground masks ``[H, W]`` aligned with ``features``.
        prepare: Whether to apply ``CrowdSegmenter_CT_GCE.metric.prepare_features``
            before computing Eq. (8) prototypes (set ``False`` if features are
            already prepared).

    Returns:
        Dictionary containing ``per_annotator`` (list of ``C_a``), ``panel_mean``
        (r10 primary panel score), ``panel_q10`` (r10 lower-decile early alarm),
        and ``panel_median``.
    """
    prepared = [prepare_features(f) if prepare else np.asarray(f, dtype=np.float32) for f in features]
    scores: List[float] = []
    for masks_a in annotator_masks:
        if len(masks_a) != len(prepared):
            raise ValueError(
                f"Expected {len(prepared)} masks per annotator, got {len(masks_a)}."
            )
        protos = [nlc_prototype(f, m) for f, m in zip(prepared, masks_a)]
        c_a, _ = nlc_consistency(protos)
        scores.append(c_a)

    return {
        "per_annotator": scores,
        "panel_mean": panel_mean(scores),
        "panel_q10": panel_lower_quantile(scores, q=0.1),
        "panel_median": panel_median(scores),
    }


class _MultiScaleResNet34Features(nn.Module):
    """Frozen multi-scale ResNet-34 extractor (layer2 + layer3) prepared per Eq. (8)."""

    def __init__(self, target_size: Tuple[int, int], pretrained: bool = True) -> None:
        super().__init__()
        from crowdsegmenter.models.backbone import ResNet34Encoder

        self.encoder = ResNet34Encoder(pretrained=pretrained, in_channels=3)
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, skips = self.encoder(x)
        l2, l3 = skips[2], skips[3]
        l2_up = F.interpolate(l2, size=self.target_size, mode="bilinear", align_corners=False)
        l3_up = F.interpolate(l3, size=self.target_size, mode="bilinear", align_corners=False)
        feat = torch.cat([l2_up, l3_up], dim=1)
        feat = feat - feat.mean(dim=(2, 3), keepdim=True)
        feat = F.normalize(feat, p=2, dim=1, eps=1e-8)
        feat = feat - feat.mean(dim=(2, 3), keepdim=True)
        return feat


@torch.no_grad()
def extract_prepared_dense_features(
    loader: DataLoader,
    device: Union[str, torch.device],
    feat_size: Optional[Tuple[int, int]] = None,
    pretrained: bool = True,
) -> List[np.ndarray]:
    """Extract Eq. (8)-prepared ResNet-34 feature maps once across ``loader``."""
    extractor: Optional[_MultiScaleResNet34Features] = None
    features: List[np.ndarray] = []

    for batch in loader:
        images = batch[0].to(device)
        if extractor is None:
            h, w = int(images.shape[2]), int(images.shape[3])
            target_size = feat_size if feat_size is not None else (min(h, 128), min(w, 128))
            extractor = _MultiScaleResNet34Features(
                target_size=target_size,
                pretrained=pretrained,
            ).to(device)
            extractor.eval()

        batch_feats = extractor(images).cpu().numpy().astype(np.float32)
        for sample_feat in batch_feats:
            features.append(sample_feat)

    return features


def _resize_mask_nearest(mask: torch.Tensor, target_size: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize a binary ``[H, W]`` tensor to ``target_size``."""
    if tuple(mask.shape) == target_size:
        return mask.float().cpu().numpy()
    resized = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0).float(),
        size=target_size,
        mode="nearest",
    )
    return resized.squeeze(0).squeeze(0).cpu().numpy()


def compute_prediction_nlc_metrics(
    features: Sequence[np.ndarray],
    seg_preds: torch.Tensor,
    num_classes: int,
    threshold: float = 0.5,
    prepare: bool = False,
) -> Dict[str, Any]:
    """Compute zero-GT NLC consistency on model predictions across foreground classes."""
    if len(features) < 2:
        raise ValueError(f"NLC requires at least 2 samples; got {len(features)}.")
    if seg_preds.shape[0] != len(features):
        raise ValueError(
            f"Prediction count ({seg_preds.shape[0]}) does not match features ({len(features)})."
        )

    prepared = [
        prepare_features(f) if prepare else np.asarray(f, dtype=np.float32) for f in features
    ]
    grid_size = (int(prepared[0].shape[1]), int(prepared[0].shape[2]))
    pred_labels = seg_preds.argmax(dim=1) if num_classes > 2 else None

    per_class: Dict[str, float] = {}
    for cls_idx in range(1, num_classes):
        if pred_labels is not None:
            class_masks = (pred_labels == cls_idx).float()
        else:
            class_masks = (seg_preds[:, cls_idx] > threshold).float()

        protos = [
            nlc_prototype(feat, _resize_mask_nearest(mask, grid_size))
            for feat, mask in zip(prepared, class_masks)
        ]
        score, _ = nlc_consistency(protos)
        per_class[f"class_{cls_idx}"] = float(score)

    macro_avg = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return {
        "per_class": per_class,
        "macro_avg": macro_avg,
    }


def compute_loader_panel_nlc_metrics(
    features: Sequence[np.ndarray],
    loader: DataLoader,
    num_classes: int,
    num_annotators: int,
    ignored_value: float = 0.6,
    prepare: bool = False,
) -> Dict[str, Any]:
    """Compute zero-GT NLC panel metrics across annotators in a multi-annotator loader."""
    prepared = [
        prepare_features(f) if prepare else np.asarray(f, dtype=np.float32) for f in features
    ]
    grid_size = (int(prepared[0].shape[1]), int(prepared[0].shape[2]))
    all_masks = torch.cat([batch[1].cpu() for batch in loader], dim=0)
    if all_masks.shape[0] != len(prepared):
        raise ValueError(
            f"Mask count ({all_masks.shape[0]}) does not match features ({len(prepared)})."
        )

    per_class: Dict[str, Dict[str, Any]] = {}
    for cls_idx in range(1, num_classes):
        ann_masks: List[List[np.ndarray]] = []
        for ann_idx in range(num_annotators):
            ch_idx = cls_idx * num_annotators + ann_idx
            ch = all_masks[:, ch_idx]
            valid = ch != ignored_value
            bin_ch = torch.where(valid, (ch > 0.5).float(), torch.zeros_like(ch))
            ann_masks.append(
                [_resize_mask_nearest(m, grid_size) for m in bin_ch]
            )
        per_class[f"class_{cls_idx}"] = compute_nlc_panel_metrics(
            features=prepared,
            annotator_masks=ann_masks,
            prepare=False,
        )

    macro_mean = (
        float(np.mean([v["panel_mean"] for v in per_class.values()]))
        if per_class
        else 0.0
    )
    return {
        "per_class": per_class,
        "macro_panel_mean": macro_mean,
    }
