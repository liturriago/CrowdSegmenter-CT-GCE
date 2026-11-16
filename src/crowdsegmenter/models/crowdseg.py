"""CrowdSeg architecture with ResNet-34 backbone-based ResUNet and AnnRelCM head."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from crowdsegmenter.config import ModelConfig
from crowdsegmenter.models.backbone import (
    Decoder,
    ResNet34Encoder,
    SegmentationHead,
    kernel_initializer,
)


class ConvBlockAnnRelCM(nn.Module):
    """Convolutional feature extractor for the AnnRelCM annotator head."""

    def __init__(self, in_channels: int, image_size: int) -> None:
        super().__init__()
        pool_spatial_size = max(image_size // 16, 1)

        self.conv = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_bn = nn.BatchNorm2d(8)
        self.conv_bn2 = nn.BatchNorm2d(4)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((pool_spatial_size, pool_spatial_size))
        self.flatten = nn.Flatten()

        self.fc_bn = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(4 * (pool_spatial_size ** 2), 128)
        self.fc2 = nn.Linear(128, 64)

        self.apply(kernel_initializer(42))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv_bn(self.conv(x))))
        x = self.pool(self.relu(self.conv_bn2(self.conv2(x))))
        x = self.pool(self.relu(self.conv_bn2(self.conv3(x))))
        x = self.pool(self.relu(self.conv_bn2(self.conv3(x))))

        x = self.adaptive_pool(x)
        x = self.flatten(x)

        x = self.relu(self.fc_bn(self.fc1(x)))
        return self.fc2(x)


class AnnRelCM(nn.Module):
    """Annotator Reliability Confusion Matrix Module predicting ``[B, K^2, H, W]``."""

    def __init__(
        self,
        num_classes: int,
        num_annotators: int,
        image_size: int,
        in_channels: int,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_annotators = num_annotators
        self.image_size = image_size

        self.conv_layers = ConvBlockAnnRelCM(in_channels, image_size)
        self.dense_annotator = nn.Linear(num_annotators, 64)
        self.dense_output = nn.Linear(128, num_classes ** 2)
        self.norm = nn.BatchNorm1d(num_classes ** 2)
        self.act = nn.Softplus()

        self.dense_annotator.apply(kernel_initializer(42))
        self.dense_output.apply(kernel_initializer(42))

    def forward(
        self,
        x: torch.Tensor,
        onehot: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        h, w = target_size if target_size is not None else (self.image_size, self.image_size)
        a_feat = self.dense_annotator(onehot)
        v_feat = self.conv_layers(x)
        output = self.dense_output(torch.hstack((a_feat, v_feat)))
        output = self.norm(output)
        output = self.act(output.view(-1, self.num_classes, self.num_classes))

        all_weights = output.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, h, w)
        return all_weights.reshape(-1, self.num_classes ** 2, h, w)


class CrowdSeg(nn.Module):
    """CrowdSeg architecture with ResNet-34 ResUNet and AnnRelCM head."""

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
        self.annotator_head = AnnRelCM(
            num_classes=config.num_classes,
            num_annotators=config.num_annotators,
            image_size=config.image_size,
            in_channels=config.decoder_channels[-1],
        )

    def forward(
        self,
        x: torch.Tensor,
        onehot: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_size = (x.shape[2], x.shape[3])
        bottleneck, skip_connections = self.encoder(x)
        decoded = self.decoder(bottleneck, skip_connections, x)
        output = self.segmentation_head(decoded, input_size)

        if onehot is not None:
            annotator_output = self.annotator_head(decoded, onehot, target_size=input_size)
            return output, annotator_output

        return output, None
