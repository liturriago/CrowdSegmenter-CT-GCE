"""CT-GCE segmentation architecture with category-indexed reliability head.

Implements the two-branch parameterization required by CT-GCE (r10 Eq. 1 & Eq. 7):
    - Segmentation branch (theta): ResNet34Encoder + Decoder + SegmentationHead (softmax)
      producing class probabilities ``P`` of shape ``[B, K, H, W]``.
    - Reliability branch (phi): CategoryReliabilityHead producing per-annotator,
      per-category reliability fields ``Lambda_r^k[w, h]`` of shape
      ``[B, R, K, H, W]`` in ``[0, 1]``.

Per r10 § "The loss", the reliability branch ``phi`` operates on detached
decoder features when trained under C3 (Eq. 7) so segmentation gradients never
mix with reliability optimization.
"""

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


class CategoryReliabilityHead(nn.Module):
    """Predicts category-indexed annotator reliability ``Lambda`` of shape ``[B, R, K, H, W]``."""

    def __init__(
        self,
        in_channels: int,
        num_annotators: int,
        num_classes: int,
        activation: Optional[str] = "sigmoid",
    ) -> None:
        super().__init__()
        self.num_annotators = num_annotators
        self.num_classes = num_classes

        self.attention_gen = nn.Sequential(
            nn.Linear(num_annotators, 128),
            nn.ReLU(),
            nn.Linear(128, in_channels * 2),
        )
        self.reliability_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_annotators * num_classes, kernel_size=1),
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
        out = self.reliability_conv(modulated)  # [B, R*K, h, w]
        out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)

        if self.activation is not None:
            out = self.activation(out)

        return out.view(b, self.num_annotators, self.num_classes, target_size[0], target_size[1])


class CTGCEModel(nn.Module):
    """CT-GCE ResUNet model outputting ``(P [B, K, H, W], Lambda [B, R, K, H, W])``."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.seg_head_activation != "softmax":
            raise ValueError(
                "CTGCEModel requires seg_head_activation='softmax' so P lies on the "
                f"probability simplex; got {config.seg_head_activation!r}."
            )

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
            activation="softmax",
        )
        self.annotator_head = CategoryReliabilityHead(
            in_channels=config.decoder_channels[-1],
            num_annotators=config.num_annotators,
            num_classes=config.num_classes,
            activation=config.annotator_activation or "sigmoid",
        )

    def forward(
        self,
        x: torch.Tensor,
        multihot: Optional[torch.Tensor] = None,
        detach_reliability_features: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through CT-GCE.

        Args:
            x: Input image tensor ``[B, C_in, H, W]``.
            multihot: Optional annotator presence vector ``[B, R]``.
            detach_reliability_features: If ``True`` (default per r10), detaches
                decoder features entering the reliability head so C3 branch
                gradients do not update the shared segmentation backbone.
        """
        input_size = (x.shape[2], x.shape[3])
        bottleneck, skip_connections = self.encoder(x)
        decoded = self.decoder(bottleneck, skip_connections, x)
        seg_probs = self.segmentation_head(decoded, input_size)

        if multihot is not None:
            rel_input = decoded.detach() if detach_reliability_features else decoded
            lambda_field = self.annotator_head(rel_input, multihot, input_size)
            return seg_probs, lambda_field

        return seg_probs, None
