"""AnnotHarmony architecture with ResNet-34 ResUNet and FiLM reliability head."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from crowdsegmenter.config import ModelConfig
from crowdsegmenter.models.backbone import (
    Decoder,
    ResNet34Encoder,
    SegmentationHead,
    get_activation,
)


class AttentionAnnRel(nn.Module):
    """Annotator-specific FiLM modulation predicting per-annotator reliability ``[B, A, H, W]``."""

    def __init__(
        self,
        in_channels: int,
        num_annotators: int,
        activation: Optional[str] = "sigmoid",
    ) -> None:
        super().__init__()
        self.attention_gen = nn.Sequential(
            nn.Linear(num_annotators, 128),
            nn.ReLU(),
            nn.Linear(128, in_channels * 2),
        )
        self.confidence_head = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_annotators, 1),
        )
        self.activation = get_activation(activation)

    def forward(
        self,
        x: torch.Tensor,
        multihot: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        params = self.attention_gen(multihot)
        gamma, beta = torch.chunk(params, 2, dim=1)
        b, c = gamma.shape

        modulated = x * gamma.view(b, c, 1, 1) + beta.view(b, c, 1, 1)
        out = self.confidence_head(modulated)
        out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)

        if self.activation is not None:
            out = self.activation(out)
        return out


class AnnotHarmony(nn.Module):
    """AnnotHarmony segmentation model with ResNet-34 backbone."""

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
        self.annotator_head = AttentionAnnRel(
            in_channels=config.decoder_channels[-1],
            num_annotators=config.num_annotators,
            activation=config.annotator_activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        multihot: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_size = (x.shape[2], x.shape[3])
        bottleneck, skip_connections = self.encoder(x)
        decoded = self.decoder(bottleneck, skip_connections, x)
        output = self.segmentation_head(decoded, input_size)

        if multihot is not None:
            annotator_output = self.annotator_head(decoded, multihot, input_size)
            return output, annotator_output

        return output, None
