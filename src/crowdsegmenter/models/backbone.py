"""Shared ResNet-34 encoder, ResUNet decoder, and segmentation head building blocks."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import resnet34 as _tv_resnet34
    _HAS_TORCHVISION = True
except ImportError:
    _tv_resnet34 = None
    _HAS_TORCHVISION = False


def kernel_initializer(seed: int) -> Callable[[nn.Module], None]:
    """Xavier uniform initializer with fixed seed for reproducibility."""

    def init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            torch.manual_seed(seed)
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    return init_weights


def get_activation(activation: Optional[str]) -> Optional[nn.Module]:
    """Return an activation module by name ('sigmoid', 'softmax', 'tanh', 'relu', or None)."""
    if activation is None:
        return None
    if isinstance(activation, str):
        key = activation.lower()
        mapping: dict[str, nn.Module] = {
            "sigmoid": nn.Sigmoid(),
            "softmax": nn.Softmax(dim=1),
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
        }
        if key in mapping:
            return mapping[key]
        raise ValueError(f"Unsupported activation: '{activation}'.")
    raise ValueError("activation must be None or a string.")


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with optional 1x1 channel-projection skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)
        self.apply(kernel_initializer(seed))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv2(self.conv1(x))
        return self.relu(out + identity)


class _BasicBlock(nn.Module):
    """Standard ResNet BasicBlock fallback used when torchvision is unavailable."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
            if stride != 1 or in_planes != planes
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


def _make_resnet34_layer(in_planes: int, planes: int, blocks: int, stride: int) -> nn.Sequential:
    layers = [_BasicBlock(in_planes, planes, stride=stride)]
    for _ in range(1, blocks):
        layers.append(_BasicBlock(planes, planes, stride=1))
    return nn.Sequential(*layers)


class ResNet34Encoder(nn.Module):
    """ResNet-34 encoder extracting multi-scale skip connections."""

    def __init__(self, pretrained: bool = True, in_channels: int = 3) -> None:
        super().__init__()

        if _HAS_TORCHVISION:
            resnet = _tv_resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
        else:
            if pretrained:
                raise ImportError(
                    "torchvision is required when pretrained=True. "
                    "Install optional 'training' extras or set pretrained=False."
                )
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = _make_resnet34_layer(64, 64, 3, stride=1)
            self.layer2 = _make_resnet34_layer(64, 128, 4, stride=2)
            self.layer3 = _make_resnet34_layer(128, 256, 6, stride=2)
            self.layer4 = _make_resnet34_layer(256, 512, 3, stride=2)

        if in_channels != 3:
            self.layer0[0] = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.out_channels = [64, 64, 128, 256, 512]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips: List[torch.Tensor] = []

        x = self.layer0(x)       # [B,  64, H/2,  W/2]
        skips.append(x)

        x = self.maxpool(x)
        x = self.layer1(x)       # [B,  64, H/4,  W/4]
        skips.append(x)

        x = self.layer2(x)       # [B, 128, H/8,  W/8]
        skips.append(x)

        x = self.layer3(x)       # [B, 256, H/16, W/16]
        skips.append(x)

        x = self.layer4(x)       # [B, 512, H/32, W/32]
        return x, skips


class Decoder(nn.Module):
    """ResUNet decoder with skip connections and final original-image skip."""

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int] = [256, 128, 64, 64],
        in_channels: int = 3,
        use_residual: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual

        skip_channels = encoder_channels[:-1][::-1] + [in_channels]
        self.decoder_blocks = nn.ModuleList()
        self.decoder_blocks.append(
            self._make_block(encoder_channels[-1], decoder_channels[0], seed)
        )
        for i in range(1, len(decoder_channels)):
            in_ch = decoder_channels[i - 1] + skip_channels[i - 1]
            self.decoder_blocks.append(self._make_block(in_ch, decoder_channels[i], seed))

    def _make_block(self, in_channels: int, out_channels: int, seed: int) -> nn.Module:
        if self.use_residual:
            return nn.Sequential(
                ResidualBlock(in_channels, out_channels, seed=seed),
                ConvBlock(out_channels, out_channels),
            )
        return nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels),
        )

    def forward(
        self,
        bottleneck: torch.Tensor,
        skip_connections: List[torch.Tensor],
        x_in: torch.Tensor,
    ) -> torch.Tensor:
        skips = skip_connections[::-1] + [x_in]
        x = F.interpolate(bottleneck, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.decoder_blocks[0](x)

        for i in range(1, len(self.decoder_blocks)):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            skip = skips[i - 1]
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.decoder_blocks[i](x)

        return x


class SegmentationHead(nn.Module):
    """1x1 conv -> bilinear upsample -> optional activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        x = self.conv(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        if self.activation is not None:
            x = self.activation(x)
        return x
