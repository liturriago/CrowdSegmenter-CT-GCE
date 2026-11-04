"""Material for the metric pilot: ported from the first prototype, unmodified.

Everything here implements no equation of the bound revision — the equations
live in `CrowdSegmenter_CT_GCE.metric` — so this module carries no
`__provenance__`. It is the pilot's material:

- `load_inria` — the prototype's Inria loader (15 paired patches, 1024x1024);
- `stable_seed` + `AnnotatorPool` — the annotator simulator. The prototype's
  seed mixed `hash(image_id)`, which Python randomizes per process, so the
  prototype's cached masks are the only stable source for P1/P2/P4/P5 and are
  read verbatim from `data/annotators/`; masks NOT in the cache (the P3 pool
  beyond annotator 0) are generated under `stable_seed`, a declared
  crc32-based replacement that is deterministic across processes. Regenerating
  a cached profile would NOT reproduce the prototype's masks;
- `soft_probability_consensus`, `majority_voting_consensus` — the prototype's
  consensus aggregation (the panel-consensus unit's object and the supervised
  reference), ported line for line;
- `compute_iou`, `compute_miou`, `compute_dice`,
  `evaluate_segmentation_batch` — supervised quality against the dataset's
  ground truth. The pilot uses these ONLY as the validation column: they never
  enter a zero-GT decision metric;
- `FeatureExtractor` — the prototype's ResNet-34 multi-scale feature
  extraction (layer2 + layer3, both bilinearly upsampled to 256x256,
  concatenated to 384 channels), forced to CPU for bitwise reproducibility
  (the prototype's own device is unknown and not load-bearing: every arm of
  the pilot is measured on the same device). The ImageNet weights are
  downloaded once by the operator and read from the torch hub cache.

Each ported function keeps the prototype's docstring facts where they still
hold; anything that changed is named in its own docstring.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, map_coordinates

_RESIZE = (256, 256)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_inria(data_dir: Union[str, Path]) -> Tuple[Dict[str, Image.Image], Dict[str, np.ndarray]]:
    """The prototype's loader: paired image/mask patches from `data_dir`.

    Images are RGB; masks are thresholded at 127 (the Inria JPEG artifacts).
    Returns ({stem: PIL.Image}, {stem: uint8 mask}).
    """
    data_dir = Path(data_dir)
    img_dir, mask_dir = data_dir / "images", data_dir / "masks"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing images or masks directory in {data_dir}")
    images: Dict[str, Image.Image] = {}
    masks: Dict[str, np.ndarray] = {}
    for img_path in sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))):
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}.jpg"
        if not mask_path.exists():
            mask_path = mask_dir / f"{stem}.png"
        if not mask_path.exists():
            continue
        images[stem] = Image.open(img_path).convert("RGB")
        masks[stem] = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)
    if not images:
        raise ValueError(f"No valid image-mask pairs found in {data_dir}")
    return images, masks


# ---------------------------------------------------------------------------
# Annotator simulation (the prototype's profiles and machinery)
# ---------------------------------------------------------------------------

def stable_seed(image_id: str, profile_key: str, annotator_id: int, base_seed: int) -> int:
    """The declared, process-stable replacement for the prototype's hash() seed.

    The prototype computed `base_seed + (hash(image_id) % 100000) + ...`, whose
    `hash()` differs between processes. Masks already in the annotator cache
    are read as files and never depend on any seed; masks generated FRESH (the
    P3 pool beyond annotator 0) use this crc32-based seed so the pilot's runs
    reproduce bitwise.
    """
    return (
        base_seed
        + (zlib.crc32(image_id.encode("utf-8")) % 100000)
        + (annotator_id * 1000)
        + (ord(profile_key[0]) * 100)
        + (int(profile_key[1]) * 10)
    ) % (2**31 - 1)


class AnnotatorPool:
    """The prototype's pool: generate-or-read one mask per (image, profile, annotator)."""

    def __init__(self, gt_masks: Dict[str, np.ndarray], cache_dir: Union[str, Path],
                 base_seed: int = 42):
        self.gt_masks = gt_masks
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_seed = base_seed

    def generate_annotator_mask(self, image_id: str, profile_key: str, annotator_id: int,
                                force_regenerate: bool = False) -> np.ndarray:
        """Read the cached mask or synthesize it with the declared stable seed."""
        cache_path = self.cache_dir / f"{profile_key}_ann{annotator_id}_{image_id}.png"
        if not force_regenerate and cache_path.exists():
            img = Image.open(cache_path).convert("L")
            return (np.array(img) > 127).astype(np.uint8)
        if image_id not in self.gt_masks:
            raise KeyError(f"no ground truth for image {image_id!r}")
        profile_seed = stable_seed(image_id, profile_key, annotator_id, self.base_seed)
        rng = np.random.RandomState(profile_seed)
        mask = _apply_smooth_elastic_deformation(
            self.gt_masks[image_id], profile_key, annotator_id, rng)
        Image.fromarray((mask * 255).astype(np.uint8)).save(cache_path)
        return mask

    def get_panel_masks(self, image_id: str, panel_profiles: List[Tuple[str, int]],
                        force_regenerate: bool = False) -> List[np.ndarray]:
        return [
            self.generate_annotator_mask(image_id, prof, ann_idx, force_regenerate)
            for prof, ann_idx in panel_profiles
        ]


def _apply_smooth_elastic_deformation(gt_mask: np.ndarray, profile_key: str,
                                      annotator_id: int, rng: np.random.RandomState) -> np.ndarray:
    """The prototype's continuous annotator noise (Simard-style elastic fields).

    Profiles, verbatim parameters and morphological biases as the prototype
    ships them; see the prototype's `annotator_simulator.py`. Only the seed
    above differs, and only for masks absent from the cache.
    """
    profiles = {
        # name: (alpha_range, sigma_range, morph_type, struct_size,
        #        n_blobs_range, blob_radius_range)
        "P1": ((0.7, 0.9), (30.0, 45.0), "none", 5, (6, 12), (30, 65)),
        "P2": ((1.8, 2.3), (25.0, 40.0), "none", 5, (6, 12), (30, 65)),
        "P3": ((1.0, 1.4), (30.0, 40.0), "p3_bias", 7, (6, 12), (30, 65)),
        "P4": ((6.5, 8.5), (25.0, 35.0), "p4_regional", 9, (6, 12), (30, 65)),
        "P5": ((14.0, 18.0), (20.0, 30.0), "p5_organic", 5, (8, 14), (30, 65)),
    }
    if profile_key not in profiles:
        raise ValueError(f"unknown profile {profile_key!r}")
    alpha_range, sigma_range, morph, struct_size, n_blobs_range, blob_range = profiles[profile_key]

    mask_bin = (gt_mask > 0.5).astype(np.float32)
    h, w = mask_bin.shape
    alpha = rng.uniform(*alpha_range)
    sigma = rng.uniform(*sigma_range)

    rx = rng.randn(h, w)
    ry = rng.randn(h, w)
    gx = gaussian_filter(rx, sigma=sigma, mode="reflect")
    gy = gaussian_filter(ry, sigma=sigma, mode="reflect")
    std_x = np.std(gx) + 1e-8
    std_y = np.std(gy) + 1e-8
    dx = alpha * (gx / std_x)
    dy = alpha * (gy / std_y)
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    warped = map_coordinates(mask_bin, (y_coords + dy, x_coords + dx), order=1, mode="nearest")
    pred = (warped >= 0.5).astype(np.float32)

    if morph == "p3_bias":
        struct = np.ones((struct_size, struct_size))
        if annotator_id % 2 == 0:
            pred = binary_dilation(pred, structure=struct).astype(np.float32)
        else:
            pred = binary_erosion(pred, structure=struct).astype(np.float32)
    elif morph == "p4_regional":
        smooth_field = gaussian_filter(rng.randn(h, w), sigma=50.0)
        smooth_field = smooth_field / (np.std(smooth_field) + 1e-8)
        struct = np.ones((struct_size, struct_size))
        dil = binary_dilation(pred, structure=struct).astype(np.float32)
        erd = binary_erosion(pred, structure=struct).astype(np.float32)
        pred = np.where(smooth_field > 1.1, dil, pred)
        pred = np.where(smooth_field < -1.1, erd, pred)
    elif morph == "p5_organic":
        num_blobs = rng.randint(*n_blobs_range)
        for _ in range(num_blobs):
            cy = rng.randint(0, h)
            cx = rng.randint(0, w)
            ry_rad = rng.randint(*blob_range)
            rx_rad = rng.randint(*blob_range)
            dist_sq = ((y_coords - cy) / ry_rad) ** 2 + ((x_coords - cx) / rx_rad) ** 2
            blob = (dist_sq <= 1.0)
            if rng.rand() > 0.35:
                pred[blob] = 1.0
            else:
                pred[blob] = 0.0
    return (pred > 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Consensus aggregation and supervised metrics (the prototype's, verbatim)
# ---------------------------------------------------------------------------

def majority_voting_consensus(masks: Union[List[np.ndarray], np.ndarray],
                              threshold: float = 0.5, seed: int = 42) -> np.ndarray:
    """Pixel-wise majority voting at 0.5 with the prototype's tie-breaking.

    Exact ties (mean vote == 0.5, possible at even panel sizes) break by 50/50
    coin flips from a seeded RandomState — the prototype's own rule, which is
    why the seed travels with the call.
    """
    if isinstance(masks, list):
        if len(masks) == 1:
            return (masks[0] > 0.5).astype(np.uint8)
        stacked = np.stack(masks, axis=0)
    else:
        if masks.shape[0] == 1:
            return (masks[0] > 0.5).astype(np.uint8)
        stacked = masks
    mean_vote = np.mean(stacked > 0.5, axis=0)
    strong_fg = mean_vote > 0.5
    strong_bg = mean_vote < 0.5
    ties = ~(strong_fg | strong_bg)
    consensus = strong_fg.astype(np.uint8)
    if np.any(ties):
        rng = np.random.RandomState(seed)
        random_flips = (rng.binomial(1, 0.5, size=mean_vote.shape) == 1)
        consensus[ties & random_flips] = 1
    return consensus


def soft_probability_consensus(masks: Union[List[np.ndarray], np.ndarray],
                               weight_by_confidence: bool = True, zero_minority: bool = True,
                               seed: int = 42) -> np.ndarray:
    """The prototype's continuous consensus (its sawtooth-free replacement).

    Minority votes (< 0.5) are zeroed, exact ties break by seeded 50/50 flips,
    and the foreground is modulated by agreement confidence:
    `base * (0.6 + 0.4 * |2*mean - 1|)` with zero_minority on — the exact rule
    the prototype's ladder ran.
    """
    if isinstance(masks, list):
        if len(masks) == 1:
            return (masks[0] > 0.5).astype(np.float32)
        stacked = np.stack(masks, axis=0).astype(np.float32)
    else:
        if masks.shape[0] == 1:
            return (masks[0] > 0.5).astype(np.float32)
        stacked = masks.astype(np.float32)
    mean_vote = np.mean(stacked > 0.5, axis=0)
    if zero_minority:
        strong_fg = mean_vote > 0.5
        strong_bg = mean_vote < 0.5
        ties = ~(strong_fg | strong_bg)
        base_mask = strong_fg.astype(np.float32)
        if np.any(ties):
            rng = np.random.RandomState(seed)
            random_flips = (rng.binomial(1, 0.5, size=mean_vote.shape) == 1)
            base_mask[ties & random_flips] = 1.0
        if not weight_by_confidence:
            return base_mask.astype(np.float32)
        agreement = np.abs(2.0 * mean_vote - 1.0)
        return (base_mask * (0.6 + 0.4 * agreement)).astype(np.float32)
    if not weight_by_confidence:
        return mean_vote.astype(np.float32)
    agreement = np.abs(2.0 * mean_vote - 1.0)
    return (mean_vote * (0.5 + 0.5 * agreement)).astype(np.float32)


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, epsilon: float = 1e-8) -> float:
    """Foreground IoU against the dataset's ground truth (validation only)."""
    p, g = pred_mask > 0.5, gt_mask > 0.5
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter / (union + epsilon))


def compute_miou(pred_mask: np.ndarray, gt_mask: np.ndarray, epsilon: float = 1e-8) -> float:
    """Two-class mean IoU (background and foreground), the prototype's definition."""
    iou_fg = compute_iou(pred_mask, gt_mask, epsilon=epsilon)
    p_bg, g_bg = pred_mask <= 0.5, gt_mask <= 0.5
    inter_bg = np.logical_and(p_bg, g_bg).sum()
    union_bg = np.logical_or(p_bg, g_bg).sum()
    return float(0.5 * (iou_fg + inter_bg / (union_bg + epsilon)))


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray, epsilon: float = 1e-8) -> float:
    """Foreground Dice against the dataset's ground truth (validation only)."""
    p, g = pred_mask > 0.5, gt_mask > 0.5
    inter = np.logical_and(p, g).sum()
    total = p.sum() + g.sum()
    return float(2.0 * inter / (total + epsilon))


def evaluate_segmentation_batch(preds: Dict[str, np.ndarray], gts: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Mean IoU / two-class mIoU / Dice of a set of predictions against GT."""
    if set(preds) != set(gts):
        raise ValueError("preds and gts must cover the same image set")
    ious = [compute_iou(preds[k], gts[k]) for k in preds]
    mious = [compute_miou(preds[k], gts[k]) for k in preds]
    dices = [compute_dice(preds[k], gts[k]) for k in preds]
    return {
        "iou": float(np.mean(ious)),
        "miou": float(np.mean(mious)),
        "dice": float(np.mean(dices)),
    }


# ---------------------------------------------------------------------------
# Feature extraction (the prototype's ResNet-34 wrapper, forced to CPU)
# ---------------------------------------------------------------------------

class _MultiScaleResNet34(torch.nn.Module):
    """The prototype's multi-scale wrapper: layer2 + layer3, both at 256x256."""

    def __init__(self, resnet) -> None:
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l2_up = F.interpolate(l2, size=_RESIZE, mode="bilinear", align_corners=False)
        l3_up = F.interpolate(l3, size=_RESIZE, mode="bilinear", align_corners=False)
        feat = torch.cat([l2_up, l3_up], dim=1)
        feat_centered = feat - feat.mean(dim=(2, 3), keepdim=True)
        feat_norm = F.normalize(feat_centered, p=2, dim=1)
        feat_norm = feat_norm - feat_norm.mean(dim=(2, 3), keepdim=True)
        return feat_norm


class FeatureExtractor:
    """The prototype's feature side, forced to CPU and read-only over images.

    The ImageNet-pretrained ResNet-34 runs in eval mode under no_grad; the
    transform is the prototype's (resize to 512x512 bilinear, ToTensor,
    ImageNet normalize). Device: CPU, declared — the prototype's device is
    unknown and no arm of the pilot depends on matching it, only on every arm
    sharing one device.
    """

    def __init__(self) -> None:
        import torchvision.models as models
        import torchvision.transforms as T

        self.device = torch.device("cpu")
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        resnet.eval()
        self.feature_extractor = _MultiScaleResNet34(resnet).to(self.device)
        self.feature_extractor.eval()
        self.image_transform = T.Compose([
            T.Resize((512, 512), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_features(self, images: Dict[str, Image.Image]) -> Dict[str, np.ndarray]:
        """One prepared feature map [384, 256, 256] per image, as numpy."""
        out: Dict[str, np.ndarray] = {}
        for img_id in sorted(images):
            tensor = self.image_transform(images[img_id]).unsqueeze(0).to(self.device)
            feat = self.feature_extractor(tensor)
            out[img_id] = feat[0].cpu().numpy()
        return out


def resize_mask_to_features(mask: np.ndarray, size: Tuple[int, int] = _RESIZE) -> np.ndarray:
    """The prototype's mask resampling: bilinear to the feature grid, re-thresholded.

    The prototype resized its annotator masks ONCE with PIL's NEAREST at load
    time for the 256-grid panels (`> 127` after resize); this helper keeps the
    same NEAREST-then-threshold rule for any mask not already at the grid size.
    """
    mask = np.asarray(mask)
    if mask.shape == size:
        return mask.astype(np.float32)
    pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(size, Image.NEAREST)
    return (np.array(pil) > 127).astype(np.float32)
