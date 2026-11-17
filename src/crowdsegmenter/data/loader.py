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

    def _load_tensor(self, path: str, is_mask: bool) -> torch.Tensor:
        target_h, target_w = int(self.config.image_size[0]), int(self.config.image_size[1])
        if not Path(path).exists():
            if is_mask:
                return torch.full(
                    (1, target_h, target_w),
                    self.config.ignored_value,
                    dtype=torch.float32,
                )
            return torch.zeros((3, target_h, target_w), dtype=torch.float32)

        with Image.open(path) as im:
            if is_mask:
                arr = np.array(im.convert("L"), dtype=np.float32)
                tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]
            else:
                arr = np.array(im.convert("RGB"), dtype=np.float32)
                tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]

        if tensor.shape[1] != target_h or tensor.shape[2] != target_w:
            mode = "nearest" if is_mask else "bilinear"
            kwargs = {"align_corners": False} if not is_mask else {}
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(target_h, target_w),
                mode=mode,
                **kwargs,
            ).squeeze(0)

        return tensor / 255.0 if self.normalize else tensor

    def __len__(self) -> int:
        return len(self.patch_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        image = self._load_tensor(self.patch_files[idx], is_mask=False)
        results: List[torch.Tensor] = [image]

        if self.config.load_annotators:
            masks, anns_ids = (
                self._load_one_annotator(idx)
                if self.mode == "CrowdSeg"
                else self._load_all_annotators(idx)
            )
            results.extend([masks, anns_ids])

        if self.config.load_ground_truth:
            gt = torch.stack(
                [
                    self._load_tensor(str(p), is_mask=True).squeeze(0)
                    for p in self.gt_masks_path[idx]
                ],
                dim=0,
            )
            results.append(gt)

        if self.transform is not None:
            if self.partition.lower() in ("train", "training"):
                results = self._apply_sync_transform(results)
            else:
                results[0] = self.transform(results[0])

        return tuple(results)

    def _load_all_annotators(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        a_count, c_count = self.config.num_annotators, self.config.num_classes
        h, w = int(self.config.image_size[0]), int(self.config.image_size[1])

        masks = torch.full((a_count * c_count, h, w), self.config.ignored_value)
        anns_ids = torch.zeros(a_count)

        for ch_idx, path in enumerate(self.masks_path[idx]):
            mask = self._load_tensor(path, is_mask=True).squeeze(0)
            masks[ch_idx] = mask
            ann_r = ch_idx % a_count
            if not torch.any(mask == self.config.ignored_value):
                anns_ids[ann_r] = 1.0

        return masks, anns_ids

    def _load_one_annotator(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        a_count, c_count = self.config.num_annotators, self.config.num_classes
        h, w = int(self.config.image_size[0]), int(self.config.image_size[1])

        masks = torch.full((a_count * c_count, h, w), self.config.ignored_value)
        anns_ids = torch.zeros(a_count)

        available = [
            r for r in range(len(self.annotators)) if Path(self.masks_path[idx][r]).exists()
        ]
        chosen_r = random.choice(available) if available else 0

        for k in range(c_count):
            ch_idx = k * a_count + chosen_r
            path = self.masks_path[idx][ch_idx]
            masks[ch_idx] = self._load_tensor(path, is_mask=True).squeeze(0)

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
        train_ops: List[Callable[[torch.Tensor], torch.Tensor]] = [_random_photometric_jitter]
        infer_ops: List[Callable[[torch.Tensor], torch.Tensor]] = []

        if self.config.mean and self.config.std:
            mean, std = self.config.mean, self.config.std
            norm_fn = lambda x: _normalize_tensor(x, mean, std)
            train_ops.append(norm_fn)
            infer_ops.append(norm_fn)

        return {
            "train": _ComposeTransforms(train_ops),
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
