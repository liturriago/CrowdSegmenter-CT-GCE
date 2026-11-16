"""Curriculum unfreezing schedules for ResNet-34 segmentation architectures."""

from __future__ import annotations

from typing import Optional, Set, Tuple

import torch.nn as nn
import torch.optim as optim

from crowdsegmenter.config import TrainConfig


def get_training_phase(
    model: nn.Module,
    config: TrainConfig,
    phase: Optional[int] = None,
    include_annotator_head: bool = True,
) -> Tuple[optim.Optimizer, Optional[optim.lr_scheduler.LRScheduler]]:
    """Apply gradual unfreezing from heads down to encoder backbone.

    Compatible with ``CrowdSeg``, ``AnnotHarmony``, and ``CTGCEModel``.
    """
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None

    if phase is None:
        params = []
        for name, param in model.named_parameters():
            if not include_annotator_head and name.startswith("annotator_head."):
                continue
            param.requires_grad = True
            params.append(param)
        optimizer = optim.Adam(params, lr=config.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
        return optimizer, scheduler

    def _freeze_all() -> None:
        for param in model.parameters():
            param.requires_grad = False

    def _unfreeze(module: nn.Module) -> list:
        params = list(module.parameters())
        for param in params:
            param.requires_grad = True
        return params

    def _unfreeze_norms(exclude: Set[nn.Parameter]) -> list:
        norm_params = []
        for m in model.encoder.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                for param in m.parameters():
                    if param not in exclude:
                        param.requires_grad = True
                        norm_params.append(param)
        return norm_params

    def _head_params() -> list:
        params = _unfreeze(model.segmentation_head)
        if include_annotator_head:
            params += _unfreeze(model.annotator_head)
        return params

    match phase:
        case 1:
            _freeze_all()
            head_params = _head_params()
            optimizer = optim.Adam([{"params": head_params, "lr": config.lr}])

        case 2:
            _freeze_all()
            head_params = _head_params()
            decoder_params = _unfreeze(model.decoder)
            optimizer = optim.Adam(
                [
                    {"params": head_params, "lr": config.lr},
                    {"params": decoder_params, "lr": config.transfer_lr},
                ]
            )

        case 3:
            _freeze_all()
            head_params = _head_params()
            decoder_params = _unfreeze(model.decoder)
            tracked = set(head_params + decoder_params)
            norm_params = _unfreeze_norms(exclude=tracked)
            optimizer = optim.Adam(
                [
                    {"params": head_params, "lr": config.lr},
                    {"params": decoder_params, "lr": config.transfer_lr},
                    {"params": norm_params, "lr": config.transfer_lr},
                ]
            )

        case 4:
            _freeze_all()
            head_params = _head_params()
            decoder_params = _unfreeze(model.decoder)
            last_block_params = _unfreeze(model.encoder.layer4)
            tracked = set(head_params + decoder_params + last_block_params)
            norm_params = _unfreeze_norms(exclude=tracked)
            optimizer = optim.Adam(
                [
                    {"params": head_params, "lr": config.lr},
                    {"params": decoder_params, "lr": config.transfer_lr},
                    {"params": norm_params, "lr": config.transfer_lr},
                    {"params": last_block_params, "lr": config.transfer_lr},
                ]
            )
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

        case 5:
            _freeze_all()
            head_params = _head_params()
            decoder_params = _unfreeze(model.decoder)
            last_block_params = _unfreeze(model.encoder.layer4)
            third_block_params = _unfreeze(model.encoder.layer3)
            tracked = set(
                head_params + decoder_params + last_block_params + third_block_params
            )
            norm_params = _unfreeze_norms(exclude=tracked)
            optimizer = optim.Adam(
                [
                    {"params": head_params, "lr": config.lr},
                    {"params": decoder_params, "lr": config.transfer_lr},
                    {"params": last_block_params, "lr": config.transfer_lr},
                    {"params": third_block_params, "lr": config.transfer_lr},
                    {"params": norm_params, "lr": config.transfer_lr},
                ]
            )
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

        case _:
            params = []
            for name, param in model.named_parameters():
                if not include_annotator_head and name.startswith("annotator_head."):
                    continue
                param.requires_grad = True
                params.append(param)
            optimizer = optim.Adam(params, lr=config.lr)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

    return optimizer, scheduler
