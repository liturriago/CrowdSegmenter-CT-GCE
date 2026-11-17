"""Training engines for multi-annotator segmentation models.

Provides:
    - ``Trainer``: Curriculum trainer for ``AnnotHarmony`` (TGCE-SSPS) and
      ``CrowdSeg`` (NoisyLabelLoss).
    - ``CTGCETrainer``: Two-phase decoupled engine for ``CTGCEModel`` (r10):
        1. Dawid-Skene noise-rate estimation ``rho_hat`` (Eq. 3).
        2. Plain crowd-GCE warm-up (no truncation).
        3. Periodic Eq. (4) cutoff calibration ``C_k = max(1/K, F_k^{-1}(rho_hat_k))``.
        4. Decoupled branch updates: ``phi`` trained by C3 (Eq. 7) with ``P``
           detached, and ``theta`` trained by CT-GCE (Eq. 1) with ``Lambda`` detached.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from crowdsegmenter.losses.ct_gce import CTGCELoss
from crowdsegmenter.utils.formatter import format_time
from crowdsegmenter.utils.metrics import MetricTracker
from crowdsegmenter.utils.train_phases import get_training_phase


class _BaseTrainer:
    """Shared evaluation, checkpointing, and phase helpers."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Any,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.use_amp = device.type == "cuda"
        self.scaler = GradScaler(device.type, enabled=self.use_amp)
        self.tracker = MetricTracker(config)
        self._use_gt: bool = bool(getattr(config, "ground_truth", False))

        self.best_val_dice: float = 0.0
        self.best_model_weights: Optional[Dict[str, torch.Tensor]] = copy.deepcopy(
            model.state_dict()
        )
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_dice": [],
            "val_dice": [],
        }
        if self._use_gt:
            self.history["val_dice_gt"] = []

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, prefix: str = "Val") -> float:
        """Evaluate probabilistic consensus metrics across ``loader``."""
        self.model.eval()
        all_seg_preds: List[torch.Tensor] = []
        all_prob_masks: List[torch.Tensor] = []

        for batch in loader:
            images = batch[0].to(self.device)
            masks = batch[1].to(self.device)
            anns_ids = batch[2].to(self.device)

            output = self.model(images, anns_ids)
            seg_pred = output[0] if isinstance(output, tuple) else output
            all_seg_preds.append(seg_pred.cpu())
            all_prob_masks.append(self.tracker.compute_probability_mask(masks).cpu())

        metrics = self.tracker.compute_probabilistic_metrics(
            torch.cat(all_seg_preds, dim=0),
            torch.cat(all_prob_masks, dim=0),
        )
        class_names = [f"Class {i}" for i in range(self.config.num_classes)]
        self.tracker.print_summary(prefix, metrics, class_names)
        return metrics["avg"]["dice"]

    @torch.no_grad()
    def evaluate_gt(self, loader: DataLoader, prefix: str = "Val (GT)") -> float:
        """Evaluate ground-truth metrics across ``loader``."""
        self.model.eval()
        all_seg_preds: List[torch.Tensor] = []
        all_gt_masks: List[torch.Tensor] = []

        for batch in loader:
            if len(batch) < 4:
                raise ValueError(
                    f"evaluate_gt requires batch length >= 4, got {len(batch)}. "
                    "Set load_ground_truth=True in DataConfig."
                )
            images = batch[0].to(self.device)
            anns_ids = batch[2].to(self.device)
            ground_truth = batch[3].to(self.device)

            output = self.model(images, anns_ids)
            seg_pred = output[0] if isinstance(output, tuple) else output
            all_seg_preds.append(seg_pred.cpu())
            all_gt_masks.append(ground_truth.cpu())

        metrics = self.tracker.compute_gt_metrics(
            torch.cat(all_seg_preds, dim=0),
            torch.cat(all_gt_masks, dim=0),
        )
        class_names = [f"Class {i}" for i in range(self.config.num_classes)]
        self.tracker.print_summary(prefix, metrics, class_names)
        return metrics["avg"]["dice"]

    def _resolve_phases(self) -> List[int]:
        phases = getattr(self.config, "epochs_phases", None)
        if isinstance(phases, list) and len(phases) == 0:
            return []
        if (
            isinstance(phases, list)
            and len(phases) >= 1
            and all(isinstance(x, (int, float)) for x in phases)
        ):
            return [int(p) for p in phases]
        return [0, 5, 10, 15]


class Trainer(_BaseTrainer):
    """Curriculum trainer for ``AnnotHarmony`` (TGCE-SSPS) and ``CrowdSeg`` (NoisyLabelLoss)."""

    _include_annotator_head: bool = True

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        config: Any,
    ) -> None:
        super().__init__(model, train_loader, val_loader, device, config)
        self.criterion = criterion
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler] = None

    def fit(self) -> Tuple[nn.Module, Dict[str, List[float]]]:
        epochs = self.config.epochs
        epochs_phases = self._resolve_phases()
        min_trace_flag = 0

        if (
            getattr(self.config, "model_name", None) == "CrowdSeg"
            and hasattr(self.criterion, "min_trace")
        ):
            # Phase 1 maximizes trace (warm-up toward identity CM); Phase 2 enables min_trace.
            self.criterion.min_trace = (
                False
                if len(epochs_phases) >= 2
                else bool(getattr(self.config, "min_trace", False))
            )

        if not epochs_phases:
            self.optimizer, self.scheduler = get_training_phase(
                self.model,
                self.config,
                phase=None,
                include_annotator_head=self._include_annotator_head,
            )

        total_train_start = time.time()

        for epoch in range(epochs):
            if epochs_phases and epoch in epochs_phases:
                phase_idx = epochs_phases.index(epoch) + 1
                self.optimizer, self.scheduler = get_training_phase(
                    self.model,
                    self.config,
                    phase=phase_idx,
                    include_annotator_head=self._include_annotator_head,
                )
                min_trace_flag += 1
                if (
                    getattr(self.config, "model_name", None) == "CrowdSeg"
                    and getattr(self.config, "min_trace", False)
                    and min_trace_flag == 2
                    and hasattr(self.criterion, "min_trace")
                ):
                    self.criterion.min_trace = True
                    print("[CrowdSeg] Phase 2: enabling trace minimization (min_trace=True)", flush=True)

            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 40)

            train_loss, train_dice, epoch_time = self._train_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["train_dice"].append(train_dice)
            print(
                f"[Train] Time: {format_time(epoch_time)} | "
                f"Loss: {train_loss:.4f} | Dice: {train_dice:.4f}"
            )

            val_dice_prob = self.evaluate(self.val_loader, prefix="Val (Prob.)")
            self.history["val_dice"].append(val_dice_prob)

            val_dice_gt = val_dice_prob
            if self._use_gt:
                val_dice_gt = self.evaluate_gt(self.val_loader, prefix="Val (GT)")
                self.history["val_dice_gt"].append(val_dice_gt)

            if self.scheduler is not None:
                self.scheduler.step()

            score = val_dice_gt if self._use_gt else val_dice_prob
            if score > self.best_val_dice:
                self.best_val_dice = score
                self.best_model_weights = copy.deepcopy(self.model.state_dict())

        print(f"\nTraining complete in {format_time(time.time() - total_train_start)}")
        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)
        return self.model, self.history

    def _train_epoch(self) -> Tuple[float, float, float]:
        if self.optimizer is None:
            raise RuntimeError("Optimizer was not initialized; check epochs_phases.")

        self.model.train()
        running_loss = 0.0
        dice_sum = 0.0
        num_batches = 0
        total_samples = 0
        epoch_start = time.time()

        for batch in self.train_loader:
            images = batch[0].to(self.device)
            masks = batch[1].to(self.device)
            anns_ids = batch[2].to(self.device)

            self.optimizer.zero_grad()
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                seg_pred, ann_pred = self.model(images, anns_ids)
                loss = self.criterion(seg_pred, ann_pred, masks, anns_ids)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            prob_mask = self.tracker.compute_probability_mask(masks)
            metrics = self.tracker.calculate_metrics(
                y_pred=seg_pred.detach(),
                y_true=prob_mask,
                threshold=self.config.threshold,
            )
            dice_sum += metrics["avg"]["dice"]
            num_batches += 1

        epoch_time = time.time() - epoch_start
        return (
            running_loss / max(total_samples, 1),
            dice_sum / max(num_batches, 1),
            epoch_time,
        )


class SupervisedTrainer(Trainer):
    """Curriculum trainer for the supervised Ground-Truth ResUNet oracle."""

    _include_annotator_head: bool = False

    def _train_epoch(self) -> Tuple[float, float, float]:
        if self.optimizer is None:
            raise RuntimeError("Optimizer was not initialized; check epochs_phases.")

        self.model.train()
        running_loss = 0.0
        dice_sum = 0.0
        num_batches = 0
        total_samples = 0
        epoch_start = time.time()

        for batch in self.train_loader:
            if len(batch) < 4:
                raise ValueError(
                    f"SupervisedTrainer requires ground_truth in batch (len={len(batch)} < 4). "
                    "Set load_ground_truth=True in DataConfig."
                )
            images = batch[0].to(self.device)
            ground_truth = batch[3].to(self.device)

            self.optimizer.zero_grad()
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                seg_pred = self.model(images)
                loss = self.criterion(seg_pred, ground_truth)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            metrics = self.tracker.calculate_metrics(
                y_pred=seg_pred.detach(),
                y_true=ground_truth,
                threshold=self.config.threshold,
            )
            dice_sum += metrics["avg"]["dice"]
            num_batches += 1

        epoch_time = time.time() - epoch_start
        return (
            running_loss / max(total_samples, 1),
            dice_sum / max(num_batches, 1),
            epoch_time,
        )


class CTGCETrainer(_BaseTrainer):
    """Decoupled CT-GCE (r10) training engine.

    Separates segmentation optimization (Eq. 1 ``ctgce_loss``) from reliability
    branch optimization (Eq. 7 ``c3_loss``) and calibrates per-category
    cutoffs ``C_k`` via Dawid-Skene ``rho_hat`` (Eqs. 3-4) after warm-up.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: CTGCELoss,
        device: torch.device,
        config: Any,
        precomputed_rho_hat: Optional[torch.Tensor] = None,
        max_ds_pixels: int = 250_000,
        max_calib_pixels: int = 500_000,
    ) -> None:
        super().__init__(model, train_loader, val_loader, device, config)
        self.criterion = criterion
        self.rho_hat: Optional[torch.Tensor] = (
            precomputed_rho_hat.clone() if precomputed_rho_hat is not None else None
        )
        self.max_ds_pixels = max_ds_pixels
        self.max_calib_pixels = max_calib_pixels
        self.cutoffs: Optional[torch.Tensor] = None
        self.history["branch_loss"] = []

        self.seg_optimizer: Optional[optim.Optimizer] = None
        self.seg_scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
        self.branch_optimizer = optim.Adam(
            self.model.annotator_head.parameters(),
            lr=getattr(config, "branch_lr", config.lr),
        )

    @staticmethod
    def _stride_batch_masks(masks: torch.Tensor, target_pixels: int) -> torch.Tensor:
        """Flatten ``[B, C, H, W]`` to ``[N_sub, C, 1, 1]`` with deterministic stride."""
        b, c, h, w = masks.shape
        flat = masks.permute(0, 2, 3, 1).reshape(-1, c)
        step = max(1, flat.shape[0] // max(target_pixels, 1))
        return flat[::step].unsqueeze(-1).unsqueeze(-1).cpu()

    @torch.no_grad()
    def _estimate_dataset_noise_rates(self) -> torch.Tensor:
        """Estimate global ``rho_hat [K]`` over the training split using Dawid-Skene EM."""
        t0 = time.time()
        print("[Dawid-Skene] Estimating per-category noise rates rho_hat...", flush=True)
        n_batches = max(len(self.train_loader), 1)
        per_batch = max(1, self.max_ds_pixels // n_batches)
        sampled_masks = [
            self._stride_batch_masks(batch[1], per_batch) for batch in self.train_loader
        ]
        stacked = torch.cat(sampled_masks, dim=0)
        rho = self.criterion.estimate_noise_rates(stacked, max_pixels=self.max_ds_pixels)
        if not bool(torch.isfinite(rho).all()):
            # Fallback to 0.0 only for unsupported categories if a small subset lacks a class
            rho = torch.nan_to_num(rho, nan=0.0)
        print(
            f"[Dawid-Skene] rho_hat = {[round(float(v), 4) for v in rho]} "
            f"({time.time() - t0:.2f}s)",
            flush=True,
        )
        return rho

    @torch.no_grad()
    def _refresh_cutoffs(self) -> torch.Tensor:
        """Calibrate Eq. (4) per-category cutoffs across the training split."""
        if self.rho_hat is None:
            self.rho_hat = self._estimate_dataset_noise_rates()

        t0 = time.time()
        self.model.eval()
        n_batches = max(len(self.train_loader), 1)
        per_batch = max(1, self.max_calib_pixels // n_batches)
        all_preds: List[torch.Tensor] = []
        all_masks: List[torch.Tensor] = []
        for batch in self.train_loader:
            images = batch[0].to(self.device)
            masks = batch[1]
            seg_pred, _ = self.model(images, multihot=None)
            all_preds.append(self._stride_batch_masks(seg_pred, per_batch))
            all_masks.append(self._stride_batch_masks(masks, per_batch))

        calib = self.criterion.calibrate(
            torch.cat(all_preds, dim=0),
            torch.cat(all_masks, dim=0),
            self.rho_hat,
            max_pixels=self.max_calib_pixels,
        )
        self.cutoffs = calib.cutoffs.to(self.device)
        print(
            f"[Calibration] C_k = {[round(float(v), 4) for v in self.cutoffs]} "
            f"({time.time() - t0:.2f}s)",
            flush=True,
        )
        return self.cutoffs

    def _init_seg_optimizer(self, phase: Optional[int]) -> None:
        opt, sched = get_training_phase(
            self.model, self.config, phase=phase, include_annotator_head=False
        )
        self.seg_optimizer = opt
        self.seg_scheduler = sched
        for p in self.model.annotator_head.parameters():
            p.requires_grad = True

    def fit(self) -> Tuple[nn.Module, Dict[str, List[float]]]:
        epochs = self.config.epochs
        warmup_epochs = int(getattr(self.config, "warmup_epochs", 5))
        calib_interval = int(getattr(self.config, "calibration_interval", 1))
        epochs_phases = self._resolve_phases()

        if self.rho_hat is None:
            self.rho_hat = self._estimate_dataset_noise_rates()

        if not epochs_phases:
            self._init_seg_optimizer(phase=None)

        total_train_start = time.time()

        for epoch in range(epochs):
            if epochs_phases and epoch in epochs_phases:
                phase_idx = epochs_phases.index(epoch) + 1
                self._init_seg_optimizer(phase=phase_idx)

            is_warmup = epoch < warmup_epochs
            if not is_warmup and (
                self.cutoffs is None or (epoch - warmup_epochs) % calib_interval == 0
            ):
                self._refresh_cutoffs()

            print(
                f"\nEpoch {epoch + 1}/{epochs} ({'Warm-up' if is_warmup else 'CT-GCE'})",
                flush=True,
            )
            print("-" * 40, flush=True)

            train_loss, branch_loss, train_dice, epoch_time = self._train_epoch(is_warmup)
            self.history["train_loss"].append(train_loss)
            self.history["branch_loss"].append(branch_loss)
            self.history["train_dice"].append(train_dice)
            print(
                f"[Train] Time: {format_time(epoch_time)} | "
                f"Seg Loss: {train_loss:.4f} | Branch C3: {branch_loss:.4f} | "
                f"Dice: {train_dice:.4f}",
                flush=True,
            )

            val_dice_prob = self.evaluate(self.val_loader, prefix="Val (Prob.)")
            self.history["val_dice"].append(val_dice_prob)

            val_dice_gt = val_dice_prob
            if self._use_gt:
                val_dice_gt = self.evaluate_gt(self.val_loader, prefix="Val (GT)")
                self.history["val_dice_gt"].append(val_dice_gt)

            if self.seg_scheduler is not None:
                self.seg_scheduler.step()

            score = val_dice_gt if self._use_gt else val_dice_prob
            if score > self.best_val_dice:
                self.best_val_dice = score
                self.best_model_weights = copy.deepcopy(self.model.state_dict())

        print(f"\nTraining complete in {format_time(time.time() - total_train_start)}")
        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)
        return self.model, self.history

    def _train_epoch(self, is_warmup: bool) -> Tuple[float, float, float, float]:
        if self.seg_optimizer is None:
            raise RuntimeError("Segmentation optimizer was not initialized.")

        self.model.train()
        seg_loss_sum = 0.0
        branch_loss_sum = 0.0
        dice_sum = 0.0
        num_batches = 0
        total_samples = 0
        epoch_start = time.time()

        for batch in self.train_loader:
            images = batch[0].to(self.device)
            masks = batch[1].to(self.device)
            anns_ids = batch[2].to(self.device)
            batch_size = images.size(0)

            # 1. Reliability branch step (phi) via Eq. (7) C3 with P detached
            branch_loss_val = 0.0
            if not is_warmup:
                self.branch_optimizer.zero_grad()
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    seg_pred_det, lambda_pred = self.model(
                        images, anns_ids, detach_reliability_features=True
                    )
                    b_loss = self.criterion.reliability_loss(
                        seg_pred_det.detach(), lambda_pred, masks
                    )
                self.scaler.scale(b_loss).backward()
                self.scaler.step(self.branch_optimizer)
                self.scaler.update()
                branch_loss_val = float(b_loss.item())

            # 2. Segmentation step (theta) via warm-up GCE or Eq. (1) CT-GCE
            self.seg_optimizer.zero_grad()
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                if is_warmup:
                    seg_pred, _ = self.model(images, multihot=None)
                    s_loss = self.criterion.warmup_loss(seg_pred, masks)
                else:
                    seg_pred, lambda_pred = self.model(
                        images, anns_ids, detach_reliability_features=True
                    )
                    assert self.cutoffs is not None and lambda_pred is not None
                    s_loss = self.criterion.segmentation_loss(
                        seg_pred, lambda_pred.detach(), masks, self.cutoffs
                    )

            self.scaler.scale(s_loss).backward()
            self.scaler.step(self.seg_optimizer)
            self.scaler.update()

            seg_loss_sum += float(s_loss.item()) * batch_size
            branch_loss_sum += branch_loss_val * batch_size
            total_samples += batch_size

            prob_mask = self.tracker.compute_probability_mask(masks)
            metrics = self.tracker.calculate_metrics(
                y_pred=seg_pred.detach(),
                y_true=prob_mask,
                threshold=self.config.threshold,
            )
            dice_sum += metrics["avg"]["dice"]
            num_batches += 1

        epoch_time = time.time() - epoch_start
        denom = max(total_samples, 1)
        return (
            seg_loss_sum / denom,
            branch_loss_sum / denom,
            dice_sum / max(num_batches, 1),
            epoch_time,
        )
