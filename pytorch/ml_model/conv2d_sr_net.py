from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_model.conv2d_block import DecoderBlock

logger = getLogger()


class ConvSrNet(nn.Module):
    def __init__(
        self,
        *,
        n_encoder_blocks: int,
        feat_channels_0: int,
        feat_channels_1: int,
        feat_channels_2: int,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        bias: bool = False,
        num_shuffle_blocks: int = 2,
        scale_factor: int = 4,
        **kwargs,
    ):
        super(ConvSrNet, self).__init__()
        assert feat_channels_1 == feat_channels_2, "Not supported yet"
        assert num_shuffle_blocks >= 1
        assert n_encoder_blocks >= 2
        assert scale_factor % 2 == 0
        assert (
            int(np.log2(scale_factor)) == num_shuffle_blocks
        ), f"{int(np.log2(scale_factor))} vs {num_shuffle_blocks}"

        logger.info(f"ConvSrNet bias = {bias}")

        self.scale_factor = scale_factor
        p = kernel_size // 2

        self.feat_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels,
                feat_channels_0,
                kernel_size=kernel_size,
                padding=p,
                bias=bias,
            ),
            nn.ReLU(),
        )

        layers = []
        for i in range(n_encoder_blocks - 1):
            layers.append(
                nn.Conv2d(
                    feat_channels_0,
                    feat_channels_0,
                    kernel_size=kernel_size,
                    padding=p,
                    bias=bias,
                )
            )
            layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)

        layers = []
        for i in range(num_shuffle_blocks):
            layers.append(
                DecoderBlock(
                    in_channels=feat_channels_0 if i == 0 else feat_channels_1,
                    out_channels=feat_channels_1 if i == 0 else feat_channels_2,
                    kernel_size=kernel_size,
                    bias=bias,
                    num_layers=2,
                )
            )

        self.last = nn.Conv2d(
            feat_channels_2,
            out_channels,
            kernel_size=kernel_size,
            padding=p,
            bias=True,
        )

        self.feat_channels = feat_channels_2

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        _x = F.interpolate(x, scale_factor=self.scale_factor, mode="bicubic")

        y = self.feat_extractor(x)
        y = self.encoder(y)
        y = self.decoder(y)

        return self.last(y) + _x

    def calc_super_resolved_features(self, x: torch.Tensor) -> torch.Tensor:
        y = self.feat_extractor(x)
        y = self.encoder(y)
        y = self.decoder(y)
        return y

    def bicubic(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale_factor, mode="bicubic")