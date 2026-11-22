"""Render ordinal segmentation predictions, annotator masks, GT, lambdas, and CMs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

from crowdsegmenter.config import ExperimentConfig
from crowdsegmenter.models.baselines import BaselineModel
from crowdsegmenter.utils.metrics import MetricTracker

# Discrete palette: -1 = Missing/Ignored (gray), 0 = Background, 1..4 = Foreground classes
_MISSING_RGB = (156, 163, 175)
_CLASS_PALETTE_RGB: Tuple[Tuple[int, int, int], ...] = (
    (26, 29, 36),    # Class 0: dark background
    (245, 158, 11),  # Class 1: amber
    (56, 189, 248),  # Class 2: sky cyan
    (52, 211, 153),  # Class 3: emerald
    (244, 114, 182), # Class 4: pink
)


@dataclass
class ModelVisualPrediction:
    """Per-sample ordinal prediction and optional interpretability maps for one model."""

    name: str
    slug: str
    pred_ordinal: np.ndarray
    pred_probs: np.ndarray
    sample_dice: Optional[float]
    lambda_field: Optional[np.ndarray] = None
    reliability_map: Optional[np.ndarray] = None
    confusion_matrices: Optional[np.ndarray] = None


@dataclass
class SampleVisualArtifact:
    """All visual inputs, targets, predictions, and model internals for one sample."""

    sample_index: int
    file_name: str
    image_rgb: np.ndarray
    annotator_ordinals: np.ndarray
    annotator_valid: np.ndarray
    gt_ordinal: Optional[np.ndarray]
    num_classes: int
    num_annotators: int
    predictions: List[ModelVisualPrediction]


def denormalize_image(
    image: torch.Tensor,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Convert normalized ``[3, H, W]`` tensor back to ``[H, W, 3]`` float32 in ``[0, 1]``."""
    img = image.detach().cpu().float().clone()
    if mean is not None and std is not None and len(mean) == img.shape[0] and len(std) == img.shape[0]:
        mean_t = torch.tensor(mean, dtype=img.dtype).view(-1, 1, 1)
        std_t = torch.tensor(std, dtype=img.dtype).view(-1, 1, 1)
        img = img * std_t + mean_t
    img = torch.clamp(img, 0.0, 1.0)
    return img.permute(1, 2, 0).numpy().astype(np.float32)


def annotator_masks_to_ordinal(
    masks: torch.Tensor,
    num_annotators: int,
    num_classes: int,
    ignored_value: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert ``[R*K, H, W]`` (channel ``k*R + r``) to ordinal ``[R, H, W]`` (``-1`` if ignored)."""
    _, h, w = masks.shape
    reshaped = masks.detach().cpu().reshape(num_classes, num_annotators, h, w).transpose(0, 1)
    ign = torch.tensor(float(ignored_value), dtype=reshaped.dtype)
    valid = ~torch.isclose(reshaped, ign, atol=1e-5).any(dim=1)
    clean = torch.where(valid.unsqueeze(1), reshaped, torch.zeros_like(reshaped))
    ordinal = torch.where(valid, clean.argmax(dim=1), torch.full_like(valid, -1, dtype=torch.long))
    return ordinal.numpy().astype(np.int64), valid.numpy().astype(bool)


def probs_to_ordinal(
    probs: torch.Tensor,
    num_classes: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert class probabilities ``[K, H, W]`` into ordinal labels ``[H, W]``."""
    p = probs.detach().cpu()
    if num_classes == 2:
        return (p[1] > threshold).long().numpy().astype(np.int64)
    return p.argmax(dim=0).numpy().astype(np.int64)


def gt_to_ordinal(gt: torch.Tensor) -> np.ndarray:
    """Convert one-hot GT tensor ``[K, H, W]`` into ordinal labels ``[H, W]``."""
    return gt.detach().cpu().argmax(dim=0).numpy().astype(np.int64)


@torch.no_grad()
def extract_crowdseg_confusion_matrices(
    model: nn.Module,
    image: torch.Tensor,
    num_annotators: int,
    num_classes: int,
    smooth: float = 1e-8,
) -> np.ndarray:
    """Evaluate CrowdSeg ``AnnRelCM`` for every annotator and column-normalize to ``[R, K, K]``."""
    img_batch = image.expand(num_annotators, -1, -1, -1)
    onehots = torch.eye(num_annotators, dtype=image.dtype, device=image.device)
    _, cms = model(img_batch, onehots)
    if cms is None:
        raise RuntimeError("CrowdSeg model returned None for confusion matrices.")
    # AnnRelCM outputs spatially constant [R, K^2, H, W]; slice (0, 0) and normalize columns
    cm_raw = cms.detach().float().reshape(num_annotators, num_classes, num_classes, cms.shape[2], cms.shape[3])
    cm_mat = cm_raw[:, :, :, 0, 0]
    cm_norm = cm_mat / (cm_mat.sum(dim=1, keepdim=True) + smooth)
    return cm_norm.cpu().numpy().astype(np.float32)


def _compute_sample_dice(
    tracker: MetricTracker,
    probs_b: torch.Tensor,
    masks_b: torch.Tensor,
    gt_b: Optional[torch.Tensor],
    threshold: float,
) -> float:
    target = gt_b if gt_b is not None else tracker.compute_probability_mask(masks_b)
    metrics = tracker.calculate_metrics(y_pred=probs_b, y_true=target, threshold=threshold)
    return float(metrics["avg"]["dice"])


@torch.no_grad()
def extract_sample_artifact(
    sample_index: int,
    file_name: str,
    sample_tuple: Tuple[torch.Tensor, ...],
    model_entries: Sequence[Tuple[Any, ExperimentConfig, nn.Module]],
    ref_cfg: ExperimentConfig,
    device: torch.device,
) -> SampleVisualArtifact:
    """Run inference for one sample across selected models and pack all visual tensors."""
    image = sample_tuple[0]
    masks = sample_tuple[1]
    anns_ids = sample_tuple[2]
    gt = sample_tuple[3] if len(sample_tuple) > 3 else None

    mean = ref_cfg.data.mean if ref_cfg.data.normalize else None
    std = ref_cfg.data.std if ref_cfg.data.normalize else None
    image_rgb = denormalize_image(image, mean=mean, std=std)
    ann_ord, ann_valid = annotator_masks_to_ordinal(
        masks,
        num_annotators=ref_cfg.model.num_annotators,
        num_classes=ref_cfg.model.num_classes,
        ignored_value=ref_cfg.data.ignored_value,
    )
    gt_ord = gt_to_ordinal(gt) if gt is not None else None

    img_b = image.unsqueeze(0).to(device)
    masks_b = masks.unsqueeze(0).to(device)
    anns_b = anns_ids.unsqueeze(0).to(device)
    gt_b = gt.unsqueeze(0).to(device) if gt is not None else None

    tracker = MetricTracker(ref_cfg.training)
    preds: List[ModelVisualPrediction] = []

    for spec, cfg, model in model_entries:
        model.eval()
        lambda_field: Optional[np.ndarray] = None
        reliability_map: Optional[np.ndarray] = None
        confusion_matrices: Optional[np.ndarray] = None

        if isinstance(model, BaselineModel):
            probs_b, _ = model(masks_b, anns_b)
        elif spec.name == "Supervised-GT":
            out = model(img_b)
            probs_b = out[0] if isinstance(out, tuple) else out
        elif spec.name == "CT-GCE":
            probs_b, lam = model(img_b, anns_b)
            if lam is not None:
                lambda_field = lam[0].detach().cpu().float().numpy()
        elif spec.name == "Annot-Harmony":
            probs_b, rel = model(img_b, anns_b)
            if rel is not None:
                reliability_map = rel[0].detach().cpu().float().numpy()
        elif spec.name == "CrowdSeg":
            probs_b, _ = model(img_b, None)
            confusion_matrices = extract_crowdseg_confusion_matrices(
                model,
                img_b,
                num_annotators=cfg.model.num_annotators,
                num_classes=cfg.model.num_classes,
                smooth=cfg.training.smooth,
            )
        else:
            probs_b, _ = model(img_b, anns_b)

        pred_ord = probs_to_ordinal(
            probs_b[0],
            num_classes=cfg.model.num_classes,
            threshold=cfg.training.threshold,
        )
        dice_val = _compute_sample_dice(
            tracker,
            probs_b,
            masks_b,
            gt_b,
            threshold=cfg.training.threshold,
        )
        preds.append(
            ModelVisualPrediction(
                name=spec.name,
                slug=spec.slug,
                pred_ordinal=pred_ord,
                pred_probs=probs_b[0].detach().cpu().float().numpy(),
                sample_dice=dice_val,
                lambda_field=lambda_field,
                reliability_map=reliability_map,
                confusion_matrices=confusion_matrices,
            )
        )

    return SampleVisualArtifact(
        sample_index=sample_index,
        file_name=file_name,
        image_rgb=image_rgb,
        annotator_ordinals=ann_ord,
        annotator_valid=ann_valid,
        gt_ordinal=gt_ord,
        num_classes=ref_cfg.model.num_classes,
        num_annotators=ref_cfg.model.num_annotators,
        predictions=preds,
    )


def _class_color_tuple(idx: int) -> Tuple[int, int, int]:
    if idx < 0:
        return _MISSING_RGB
    return _CLASS_PALETTE_RGB[idx % len(_CLASS_PALETTE_RGB)]


def _ordinal_to_rgb(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for val in np.unique(mask):
        rgb[mask == int(val)] = _class_color_tuple(int(val))
    return rgb


def _scalar_to_rgb(field: np.ndarray) -> np.ndarray:
    """Simple dark-purple-to-gold fallback colormap for PIL rendering without matplotlib."""
    v = np.clip(field.astype(np.float32), 0.0, 1.0)
    r = (v ** 0.8 * 252.0).astype(np.uint8)
    g = ((v ** 1.4) * 210.0).astype(np.uint8)
    b = (np.sin(v * np.pi) * 180.0 + 30.0).clip(0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _render_sample_pil(artifact: SampleVisualArtifact, output_path: Path) -> Path:
    """Fallback PIL composite grid renderer when matplotlib is not installed."""
    rows: List[List[Tuple[str, Image.Image]]] = []
    h, w = artifact.image_rgb.shape[:2]

    row0: List[Tuple[str, Image.Image]] = [
        ("Input", Image.fromarray((artifact.image_rgb * 255).astype(np.uint8))),
    ]
    if artifact.gt_ordinal is not None:
        row0.append(("GT", Image.fromarray(_ordinal_to_rgb(artifact.gt_ordinal))))
    for r in range(artifact.num_annotators):
        lbl = f"Ann {r}" if artifact.annotator_valid[r].any() else f"Ann {r} (missing)"
        row0.append((lbl, Image.fromarray(_ordinal_to_rgb(artifact.annotator_ordinals[r]))))
    rows.append(row0)

    row1: List[Tuple[str, Image.Image]] = []
    for pred in artifact.predictions:
        title = f"{pred.name} ({pred.sample_dice:.3f})" if pred.sample_dice is not None else pred.name
        row1.append((title, Image.fromarray(_ordinal_to_rgb(pred.pred_ordinal))))
    rows.append(row1)

    for pred in artifact.predictions:
        if pred.lambda_field is not None:
            for k in range(artifact.num_classes):
                lrow = [
                    (f"CT-GCE lam k={k} r={r}", Image.fromarray(_scalar_to_rgb(pred.lambda_field[r, k])))
                    for r in range(artifact.num_annotators)
                ]
                rows.append(lrow)
        if pred.reliability_map is not None:
            rrow = [
                (f"Annot-H rel r={r}", Image.fromarray(_scalar_to_rgb(pred.reliability_map[r])))
                for r in range(artifact.num_annotators)
            ]
            rows.append(rrow)
        if pred.confusion_matrices is not None:
            crow = []
            for r in range(artifact.num_annotators):
                cm = pred.confusion_matrices[r]
                up = np.kron(cm, np.ones((max(1, h // cm.shape[0]), max(1, w // cm.shape[1])), dtype=np.float32))
                im = Image.fromarray(_scalar_to_rgb(up)).resize((w, h), Image.NEAREST)
                crow.append((f"CrowdSeg CM r={r}", im))
            rows.append(crow)

    max_cols = max(len(r) for r in rows)
    pad = 8
    header_h = 20
    canvas_w = max_cols * (w + pad) + pad
    canvas_h = len(rows) * (h + header_h + pad) + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for r_idx, row in enumerate(rows):
        y0 = pad + r_idx * (h + header_h + pad)
        for c_idx, (title, im) in enumerate(row):
            x0 = pad + c_idx * (w + pad)
            draw.text((x0, y0), title[:24], fill=(20, 20, 20))
            canvas.paste(im.resize((w, h), Image.NEAREST), (x0, y0 + header_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _render_sample_matplotlib(
    artifact: SampleVisualArtifact,
    output_path: Path,
    class_names: Sequence[str],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    # Discrete ordinal cmap covering -1 (missing) and 0..K-1
    colors = [tuple(c / 255.0 for c in _MISSING_RGB)] + [
        tuple(c / 255.0 for c in _CLASS_PALETTE_RGB[k % len(_CLASS_PALETTE_RGB)])
        for k in range(artifact.num_classes)
    ]
    cmap_ord = ListedColormap(colors)
    bounds = np.arange(-1.5, artifact.num_classes + 0.5, 1.0)
    norm_ord = BoundaryNorm(bounds, cmap_ord.N)

    # Determine rows cleanly: Row 0 (data), Row 1 (preds), plus optional internals rows
    row_specs: List[Tuple[str, int]] = []
    n_data_cols = 1 + (1 if artifact.gt_ordinal is not None else 0) + artifact.num_annotators
    row_specs.append(("data", n_data_cols))
    row_specs.append(("preds", max(1, len(artifact.predictions))))

    ctgce_pred = next((p for p in artifact.predictions if p.lambda_field is not None), None)
    ah_pred = next((p for p in artifact.predictions if p.reliability_map is not None), None)
    cs_pred = next((p for p in artifact.predictions if p.confusion_matrices is not None), None)

    if ctgce_pred is not None:
        for k in range(artifact.num_classes):
            row_specs.append((f"ctgce_lambda_{k}", artifact.num_annotators))
    if ah_pred is not None:
        row_specs.append(("ah_rel", artifact.num_annotators))
    if cs_pred is not None:
        row_specs.append(("cs_cm", artifact.num_annotators))

    n_rows = len(row_specs)
    n_cols = max(c for _, c in row_specs)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.8 * n_cols, 2.75 * n_rows),
        squeeze=False,
    )
    for ax in axes.ravel():
        ax.axis("off")

    # Row 0: Input + GT + Annotators
    col = 0
    axes[0, col].imshow(artifact.image_rgb)
    axes[0, col].set_title("Input", fontsize=9, fontweight="bold")
    axes[0, col].axis("on")
    axes[0, col].set_xticks([])
    axes[0, col].set_yticks([])
    col += 1

    if artifact.gt_ordinal is not None:
        axes[0, col].imshow(artifact.gt_ordinal, cmap=cmap_ord, norm=norm_ord, interpolation="nearest")
        axes[0, col].set_title("Ground Truth (Ord.)", fontsize=9, fontweight="bold")
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        axes[0, col].axis("on")
        col += 1

    for r in range(artifact.num_annotators):
        has_ann = bool(artifact.annotator_valid[r].any())
        axes[0, col].imshow(
            artifact.annotator_ordinals[r],
            cmap=cmap_ord,
            norm=norm_ord,
            interpolation="nearest",
        )
        axes[0, col].set_title(f"Ann {r}" if has_ann else f"Ann {r} (missing)", fontsize=9)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        axes[0, col].axis("on")
        col += 1

    # Row 1: Model Ordinal Predictions
    for idx, pred in enumerate(artifact.predictions):
        axes[1, idx].imshow(
            pred.pred_ordinal,
            cmap=cmap_ord,
            norm=norm_ord,
            interpolation="nearest",
        )
        score_str = f"\nDice={pred.sample_dice:.3f}" if pred.sample_dice is not None else ""
        axes[1, idx].set_title(f"{pred.name}{score_str}", fontsize=9, fontweight="bold")
        axes[1, idx].set_xticks([])
        axes[1, idx].set_yticks([])
        axes[1, idx].axis("on")

    row_cursor = 2
    last_scalar_im = None

    if ctgce_pred is not None and ctgce_pred.lambda_field is not None:
        for k in range(artifact.num_classes):
            cname = class_names[k] if k < len(class_names) else f"Class {k}"
            for r in range(artifact.num_annotators):
                ax = axes[row_cursor, r]
                last_scalar_im = ax.imshow(
                    ctgce_pred.lambda_field[r, k],
                    cmap="magma",
                    vmin=0.0,
                    vmax=1.0,
                )
                suffix = "" if artifact.annotator_valid[r].any() else " [absent]"
                ax.set_title(f"CT-GCE $\\Lambda$ ({cname})\nAnn {r}{suffix}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis("on")
            row_cursor += 1

    if ah_pred is not None and ah_pred.reliability_map is not None:
        for r in range(artifact.num_annotators):
            ax = axes[row_cursor, r]
            last_scalar_im = ax.imshow(
                ah_pred.reliability_map[r],
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            suffix = "" if artifact.annotator_valid[r].any() else " [absent]"
            ax.set_title(f"Annot-Harmony Rel.\nAnn {r}{suffix}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("on")
        row_cursor += 1

    if cs_pred is not None and cs_pred.confusion_matrices is not None:
        k_cnt = artifact.num_classes
        for r in range(artifact.num_annotators):
            ax = axes[row_cursor, r]
            cm = cs_pred.confusion_matrices[r]
            ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
            for i in range(k_cnt):
                for j in range(k_cnt):
                    val = float(cm[i, j])
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if val > 0.55 else "black",
                    )
            ax.set_title(f"CrowdSeg CM (Ann {r})", fontsize=8)
            ax.set_xticks(range(k_cnt))
            ax.set_yticks(range(k_cnt))
            ax.tick_params(labelsize=7)
            ax.axis("on")

    legend_handles = [
        Patch(facecolor=tuple(c / 255.0 for c in _CLASS_PALETTE_RGB[k % len(_CLASS_PALETTE_RGB)]), label=class_names[k])
        for k in range(artifact.num_classes)
    ]
    if bool((~artifact.annotator_valid).any()):
        legend_handles.append(
            Patch(facecolor=tuple(c / 255.0 for c in _MISSING_RGB), label="Missing / Ignored")
        )

    fig.suptitle(
        f"Sample #{artifact.sample_index} — {artifact.file_name}",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        ncol=len(legend_handles),
        fontsize=8,
        frameon=True,
    )
    if last_scalar_im is not None:
        cbar_ax = fig.add_axes((0.93, 0.12, 0.014, 0.35))
        fig.colorbar(last_scalar_im, cax=cbar_ax, label="Reliability [0, 1]")
        fig.tight_layout(rect=(0.0, 0.0, 0.915, 0.955))
    else:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_sample_figure(
    artifact: SampleVisualArtifact,
    output_path: str | Path,
    class_names: Optional[Sequence[str]] = None,
) -> Path:
    """Save the composed prediction & interpretability figure to ``output_path``."""
    out = Path(output_path)
    names = (
        list(class_names)
        if class_names is not None
        else [f"Class {k}" for k in range(artifact.num_classes)]
    )
    try:
        return _render_sample_matplotlib(artifact, out, class_names=names)
    except ImportError:
        return _render_sample_pil(artifact, out)


def save_sample_arrays(artifact: SampleVisualArtifact, output_path: str | Path) -> Path:
    """Persist raw sample inputs, ordinal masks, predictions, lambdas, and CMs to ``.npz``."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "sample_index": np.array(artifact.sample_index, dtype=np.int64),
        "file_name": np.array(artifact.file_name),
        "image_rgb": artifact.image_rgb,
        "annotator_ordinals": artifact.annotator_ordinals,
        "annotator_valid": artifact.annotator_valid,
    }
    if artifact.gt_ordinal is not None:
        payload["gt_ordinal"] = artifact.gt_ordinal
    for pred in artifact.predictions:
        payload[f"{pred.slug}_ordinal"] = pred.pred_ordinal
        payload[f"{pred.slug}_probs"] = pred.pred_probs
        if pred.lambda_field is not None:
            payload[f"{pred.slug}_lambda"] = pred.lambda_field
        if pred.reliability_map is not None:
            payload[f"{pred.slug}_reliability"] = pred.reliability_map
        if pred.confusion_matrices is not None:
            payload[f"{pred.slug}_cm"] = pred.confusion_matrices
    np.savez_compressed(out, **payload)
    return out
