"""Multi-annotator dataset and DataLoader manager."""

from __future__ import annotations

import glob
import os
import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from crowdsegmenter.config import DataConfig


def _alphanumeric_key(text: str) -> List[Union[int, str]]:
    """Natural sort key handling embedded integers."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", text)]


class _ComposeTransforms:
    """Lightweight composition of tensor image transforms."""

    def __init__(self, ops: List[Callable[[torch.Tensor], torch.Tensor]]) -> None:
        self.ops = ops

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        for op in self.ops:
            tensor = op(tensor)
        return tensor


def _random_photometric_jitter(
    img: torch.Tensor,
    brightness: float = 0.2,
    contrast: float = 0.2,
) -> torch.Tensor:
    """Apply simple brightness/contrast jitter to a ``[C, H, W]`` float tensor in ``[0, 1]``."""
    if brightness > 0:
        b_factor = random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
        img = img * b_factor
    if contrast > 0:
        c_factor = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
        mean = img.mean(dim=(1, 2), keepdim=True)
        img = (img - mean) * c_factor + mean
    return torch.clamp(img, 0.0, 1.0)


def _random_brightness_contrast(
    img: torch.Tensor,
    brightness_limit: float = 0.15,
    contrast_limit: float = 0.15,
    p: float = 0.5,
) -> torch.Tensor:
    """Apply random brightness and contrast perturbation matching Albumentations semantics."""
    if random.random() >= p:
        return img
    c_factor = 1.0 + random.uniform(-contrast_limit, contrast_limit)
    b_shift = random.uniform(-brightness_limit, brightness_limit)
    mean = img.mean(dim=(1, 2), keepdim=True)
    return torch.clamp((img - mean) * c_factor + mean + b_shift, 0.0, 1.0)


def _rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
    """Convert ``[3, H, W]`` RGB tensor in ``[0, 1]`` to HSV in ``[0, 1]``."""
    r, g, b = img[0], img[1], img[2]
    maxc, max_idx = img.max(dim=0)
    minc = img.min(dim=0).values
    diff = maxc - minc

    hue = torch.zeros_like(maxc)
    nz = diff > 1e-6
    safe_diff = torch.where(nz, diff, torch.ones_like(diff))

    is_r = (max_idx == 0) & nz
    is_g = (max_idx == 1) & nz
    is_b = (max_idx == 2) & nz

    hue = torch.where(is_r, ((g - b) / safe_diff) % 6.0, hue)
    hue = torch.where(is_g, (b - r) / safe_diff + 2.0, hue)
    hue = torch.where(is_b, (r - g) / safe_diff + 4.0, hue)
    hue = (hue / 6.0) % 1.0

    sat = torch.where(maxc > 1e-6, diff / torch.clamp(maxc, min=1e-6), torch.zeros_like(maxc))
    return torch.stack([hue, sat, maxc], dim=0)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """Convert ``[3, H, W]`` HSV tensor in ``[0, 1]`` back to RGB in ``[0, 1]``."""
    h, s, v = hsv[0], hsv[1], hsv[2]
    h6 = (h % 1.0) * 6.0
    i = torch.floor(h6).to(torch.int64) % 6
    f = h6 - torch.floor(h6)

    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(i == 2, p, torch.where(i == 3, p, torch.where(i == 4, t, v)))))
    g = torch.where(i == 0, t, torch.where(i == 1, v, torch.where(i == 2, v, torch.where(i == 3, q, torch.where(i == 4, p, p)))))
    b = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(i == 2, t, torch.where(i == 3, v, torch.where(i == 4, v, q)))))
    return torch.clamp(torch.stack([r, g, b], dim=0), 0.0, 1.0)


def _random_hsv(
    img: torch.Tensor,
    hue_shift_limit: float = 8.0,
    sat_shift_limit: float = 15.0,
    val_shift_limit: float = 10.0,
    p: float = 0.4,
) -> torch.Tensor:
    """Apply OpenCV/Albumentations-scaled Hue/Saturation/Value jitter on ``[3, H, W]`` image."""
    if img.shape[0] != 3 or random.random() >= p:
        return img
    dh = random.uniform(-hue_shift_limit, hue_shift_limit) / 180.0
    ds = random.uniform(-sat_shift_limit, sat_shift_limit) / 255.0
    dv = random.uniform(-val_shift_limit, val_shift_limit) / 255.0

    hsv = _rgb_to_hsv(img)
    h = (hsv[0] + dh) % 1.0
    s = torch.clamp(hsv[1] + ds, 0.0, 1.0)
    v = torch.clamp(hsv[2] + dv, 0.0, 1.0)
    return _hsv_to_rgb(torch.stack([h, s, v], dim=0))


def _random_gaussian_blur(
    img: torch.Tensor,
    blur_limit: Tuple[int, int] = (3, 5),
    p: float = 0.15,
) -> torch.Tensor:
    """Apply separable Gaussian blur with odd kernel size in ``blur_limit``."""
    if random.random() >= p:
        return img
    candidates = [k for k in range(blur_limit[0], blur_limit[1] + 1) if k % 2 == 1 and k >= 3]
    if not candidates:
        return img
    ksize = random.choice(candidates)
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1.0) + 0.8
    coords = torch.arange(ksize, dtype=img.dtype, device=img.device) - (ksize - 1) / 2.0
    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()

    c, _, _ = img.shape
    pad = ksize // 2
    x = F.pad(img.unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    kx = kernel_1d.view(1, 1, 1, ksize).expand(c, 1, 1, ksize)
    ky = kernel_1d.view(1, 1, ksize, 1).expand(c, 1, ksize, 1)
    x = F.conv2d(x, kx, groups=c)
    x = F.conv2d(x, ky, groups=c)
    return torch.clamp(x.squeeze(0), 0.0, 1.0)


def _normalize_tensor(
    img: torch.Tensor,
    mean: List[float],
    std: List[float],
) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    return (img - mean_t) / std_t


class CrowdSegmenterDataset(Dataset):
    """Unified multi-annotator segmentation dataset.

    Supported modes:
        - ``"Annot-Harmony"`` / ``"CT-GCE"``: loads all annotators simultaneously
          into ``[A*C, H, W]`` (channel ``k * A + r``) and multi-hot ``anns_ids [A]``.
        - ``"CrowdSeg"``: randomly samples one valid annotator per sample into
          ``[A*C, H, W]`` (other channels filled with ``ignored_value``) and
          one-hot ``anns_ids [A]``.
    """

    MODES = ("Annot-Harmony", "CT-GCE", "CrowdSeg")

    def __init__(
        self,
        config: DataConfig,
        partition: str,
        mode: Literal["Annot-Harmony", "CT-GCE", "CrowdSeg"] = "CT-GCE",
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        cache_in_memory: bool = False,
    ) -> None:
        mode_mapping = {
            "annot-harmony": "Annot-Harmony",
            "annotharmony": "Annot-Harmony",
            "ct-gce": "CT-GCE",
            "ctgce": "CT-GCE",
            "crowdseg": "CrowdSeg",
        }
        normalized_mode = mode_mapping.get(mode.lower(), mode)
        if normalized_mode not in self.MODES:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {self.MODES}.")

        self.config = config
        self.partition = partition
        self.mode = normalized_mode
        self.transform = transform
        self.normalize = config.normalize
        self.data_path = Path(config.data_dir) / partition
        self._cache: Optional[Dict[int, Tuple[torch.Tensor, ...]]] = (
            {} if cache_in_memory else None
        )

        if not self.data_path.exists():
            raise FileNotFoundError(f"Partition directory not found: {self.data_path}")

        patch_files: List[str] = []
        for fmt in ("*.png", "*.jpg", "*.jpeg"):
            patch_files.extend(glob.glob(str(self.data_path / config.images_folder / fmt)))
        self.patch_files = sorted(patch_files, key=_alphanumeric_key)
        self.file_names = [Path(f).name for f in self.patch_files]

        self.annotators, self.masks_path, self.gt_masks_path = self._index_masks()

    def _index_masks(self) -> Tuple[List[str], List[List[str]], List[List[Path]]]:
        mask_root = self.data_path / self.config.masks_folder
        annotators = sorted(
            [
                ann
                for ann in os.listdir(mask_root)
                if os.path.isdir(mask_root / ann) and ann != self.config.ground_truth_folder
            ],
            key=_alphanumeric_key,
        )

        masks_path: List[List[str]] = []
        gt_masks_path: List[List[Path]] = []

        for sample in self.file_names:
            sample_masks: List[str] = []
            for c in range(self.config.num_classes):
                for ann in annotators:
                    sample_masks.append(str(mask_root / ann / f"class_{c}" / sample))
            masks_path.append(sample_masks)

            if self.config.load_ground_truth:
                gt_masks_path.append(
                    [
                        mask_root / self.config.ground_truth_folder / f"class_{c}" / sample
                        for c in range(self.config.num_classes)
                    ]
                )
            else:
                gt_masks_path.append([])

        return annotators, masks_path, gt_masks_path

    def _load_tensor(
        self,
        path: str,
        is_mask: bool,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if not Path(path).exists():
            fallback_h = target_size[0] if target_size is not None else int(self.config.image_size[0])
            fallback_w = target_size[1] if target_size is not None else int(self.config.image_size[1])
            if is_mask:
                return torch.full(
                    (1, fallback_h, fallback_w),
                    self.config.ignored_value,
                    dtype=torch.float32,
                )
            return torch.zeros((3, fallback_h, fallback_w), dtype=torch.float32)

        with Image.open(path) as im:
            if is_mask:
                arr = np.array(im.convert("L"), dtype=np.float32)
                tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]
            else:
                arr = np.array(im.convert("RGB"), dtype=np.float32)
                tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]

        if target_size is not None and (
            tensor.shape[1] != target_size[0] or tensor.shape[2] != target_size[1]
        ):
            mode = "nearest" if is_mask else "bilinear"
            kwargs = {"align_corners": False} if not is_mask else {}
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=target_size,
                mode=mode,
                **kwargs,
            ).squeeze(0)

        if is_mask and self.config.enh_aug:
            return (tensor > 127.0).float()
        return tensor / 255.0 if self.normalize else tensor

    def __len__(self) -> int:
        return len(self.patch_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]

        is_train = self.partition.lower() in ("train", "training")
        use_enh_train = bool(self.config.enh_aug and is_train and self.transform is not None)
        out_size = (int(self.config.image_size[0]), int(self.config.image_size[1]))

        img_target = None if use_enh_train else out_size
        image = self._load_tensor(self.patch_files[idx], is_mask=False, target_size=img_target)
        mask_target = (int(image.shape[1]), int(image.shape[2]))
        results: List[torch.Tensor] = [image]

        if self.config.load_annotators:
            masks, anns_ids = (
                self._load_one_annotator(idx, mask_target)
                if self.mode == "CrowdSeg"
                else self._load_all_annotators(idx, mask_target)
            )
            results.extend([masks, anns_ids])

        if self.config.load_ground_truth:
            gt = torch.stack(
                [
                    self._load_tensor(str(p), is_mask=True, target_size=mask_target).squeeze(0)
                    for p in self.gt_masks_path[idx]
                ],
                dim=0,
            )
            results.append(gt)

        if self.transform is not None:
            if use_enh_train:
                results = self._apply_enhanced_transform(results, out_size)
            elif is_train:
                results = self._apply_sync_transform(results)
            else:
                results[0] = self.transform(results[0])

        out = tuple(results)
        if self._cache is not None:
            self._cache[idx] = out
        return out

    def _load_all_annotators(
        self,
        idx: int,
        target_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a_count, c_count = self.config.num_annotators, self.config.num_classes
        h, w = target_size

        masks = torch.full((a_count * c_count, h, w), self.config.ignored_value)
        anns_ids = torch.zeros(a_count)

        for ch_idx, path in enumerate(self.masks_path[idx]):
            mask = self._load_tensor(path, is_mask=True, target_size=target_size).squeeze(0)
            masks[ch_idx] = mask
            ann_r = ch_idx % a_count
            if not torch.any(mask == self.config.ignored_value):
                anns_ids[ann_r] = 1.0

        return masks, anns_ids

    def _load_one_annotator(
        self,
        idx: int,
        target_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a_count, c_count = self.config.num_annotators, self.config.num_classes
        h, w = target_size

        masks = torch.full((a_count * c_count, h, w), self.config.ignored_value)
        anns_ids = torch.zeros(a_count)

        available = [
            r for r in range(len(self.annotators)) if Path(self.masks_path[idx][r]).exists()
        ]
        chosen_r = random.choice(available) if available else 0

        for k in range(c_count):
            ch_idx = k * a_count + chosen_r
            path = self.masks_path[idx][ch_idx]
            masks[ch_idx] = self._load_tensor(path, is_mask=True, target_size=target_size).squeeze(0)

        anns_ids[chosen_r] = 1.0
        return masks, anns_ids

    def _apply_sync_transform(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        if random.random() > 0.5:
            tensors = [torch.flip(t, dims=[-1]) if t.ndim > 1 else t for t in tensors]
        if random.random() > 0.5:
            tensors = [torch.flip(t, dims=[-2]) if t.ndim > 1 else t for t in tensors]
        if self.transform is not None:
            tensors[0] = self.transform(tensors[0])
        return tensors

    def _apply_enhanced_transform(
        self,
        tensors: List[torch.Tensor],
        out_size: Tuple[int, int],
    ) -> List[torch.Tensor]:
        """Apply NAIP-style anti-center-bias crop, aerial flips/rot90, resize, and radiometric jitter."""
        h, w = int(tensors[0].shape[1]), int(tensors[0].shape[2])
        crop_cfg = self.config.enh_crop_size
        if crop_cfg is not None:
            crop_h, crop_w = int(crop_cfg[0]), int(crop_cfg[1])
            if h >= crop_h and w >= crop_w and (h != crop_h or w != crop_w):
                top = random.randint(0, h - crop_h)
                left = random.randint(0, w - crop_w)
                tensors = [
                    t[:, top : top + crop_h, left : left + crop_w] if t.ndim >= 3 else t
                    for t in tensors
                ]

        if random.random() < 0.5:
            tensors = [torch.flip(t, dims=[-1]) if t.ndim >= 3 else t for t in tensors]
        if random.random() < 0.5:
            tensors = [torch.flip(t, dims=[-2]) if t.ndim >= 3 else t for t in tensors]
        if random.random() < 0.5:
            k_rot = random.randint(1, 3)
            tensors = [torch.rot90(t, k=k_rot, dims=(-2, -1)) if t.ndim >= 3 else t for t in tensors]

        cur_h, cur_w = int(tensors[0].shape[1]), int(tensors[0].shape[2])
        if (cur_h, cur_w) != out_size:
            resized: List[torch.Tensor] = [
                torch.clamp(
                    F.interpolate(
                        tensors[0].unsqueeze(0),
                        size=out_size,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0),
                    0.0,
                    1.0,
                )
            ]
            for t in tensors[1:]:
                if t.ndim >= 3:
                    resized.append(
                        F.interpolate(t.unsqueeze(0), size=out_size, mode="nearest").squeeze(0)
                    )
                else:
                    resized.append(t)
            tensors = resized

        img = tensors[0]
        img = _random_brightness_contrast(img, brightness_limit=0.15, contrast_limit=0.15, p=0.5)
        img = _random_hsv(img, hue_shift_limit=8.0, sat_shift_limit=15.0, val_shift_limit=10.0, p=0.4)
        img = _random_gaussian_blur(img, blur_limit=(3, 5), p=0.15)
        if self.transform is not None:
            img = self.transform(img)
        tensors[0] = img
        return tensors


class CrowdSegmenterDataLoader:
    """DataLoader manager creating train / val / test loaders for CrowdSegmenterDataset."""

    def __init__(
        self,
        config: DataConfig,
        mode: Literal["Annot-Harmony", "CT-GCE", "CrowdSeg"] = "CT-GCE",
    ) -> None:
        self.config = config
        self.mode = mode
        self.transforms = config.transforms
        self.partitions = config.partitions

    def _get_transforms(self) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
        train_ops: List[Callable[[torch.Tensor], torch.Tensor]] = []
        if not self.config.enh_aug:
            train_ops.append(_random_photometric_jitter)
        infer_ops: List[Callable[[torch.Tensor], torch.Tensor]] = []

        if self.config.mean and self.config.std:
            mean, std = self.config.mean, self.config.std
            norm_fn = lambda x: _normalize_tensor(x, mean, std)
            train_ops.append(norm_fn)
            infer_ops.append(norm_fn)

        return {
            "train": _ComposeTransforms(train_ops) if train_ops else (lambda x: x),
            "inference": _ComposeTransforms(infer_ops) if infer_ops else (lambda x: x),
        }

    def get_split_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Build and return ``(train_loader, val_loader, test_loader)``."""
        loaders: Dict[str, DataLoader] = {}
        tfms = self._get_transforms() if self.transforms else None

        for split in self.partitions:
            is_train = split.lower() in ("train", "training")
            transform = None
            if tfms is not None:
                transform = tfms["train"] if is_train else tfms["inference"]

            tag = split.lower()
            if "train" in tag:
                key = "train"
            elif "val" in tag:
                key = "val"
            elif "test" in tag:
                key = "test"
            else:
                key = tag

            # CrowdSeg only samples 1 annotator during training; Valid/Test must load all annotators
            # so consensus probabilistic evaluation & checkpoint selection are unbiased.
            split_mode: Literal["Annot-Harmony", "CT-GCE", "CrowdSeg"] = (
                "CT-GCE" if (self.mode == "CrowdSeg" and not is_train) else self.mode
            )
            dataset = CrowdSegmenterDataset(
                config=self.config,
                partition=split,
                mode=split_mode,
                transform=transform,
                cache_in_memory=bool(self.config.cache_val and key == "val" and not is_train),
            )

            loaders[key] = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=is_train,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                prefetch_factor=(
                    self.config.prefetch_factor if self.config.num_workers > 0 else None
                ),
                persistent_workers=self.config.num_workers > 0,
            )

        return loaders["train"], loaders["val"], loaders["test"]
