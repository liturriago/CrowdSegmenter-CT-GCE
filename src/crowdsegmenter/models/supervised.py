"""Supervised ResNet-34 ResUNet oracle baseline (no annotator reliability head)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from crowdsegmenter.config import ModelConfig
from crowdsegmenter.models.backbone import (
    Decoder,
    ResNet34Encoder,
    SegmentationHead,
)


class SupervisedResUNet(nn.Module):
    """Standard supervised ResNet-34 ResUNet sharing the exact backbone and heads."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = ResNet34Encoder(
            pretrained=config.pretrained,
            in_channels=config.in_channels,
        )
        self.decoder = Decoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=config.decoder_channels,
            in_channels=config.in_channels,
            use_residual=config.use_residual,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=config.num_classes,
            activation=config.seg_head_activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        multihot: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict class probabilities ``[B, K, H, W]`` (ignores ``multihot``)."""
        del multihot
        input_size = (x.shape[2], x.shape[3])
        bottleneck, skip_connections = self.encoder(x)
        decoded = self.decoder(bottleneck, skip_connections, x)
        return self.segmentation_head(decoded, input_size)
