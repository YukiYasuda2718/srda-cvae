import typing
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_model.conv2d_block import EncoderBlock

# Ref:
# https://github.com/unnir/cVAE/blob/baad46d96b78c4e604bb3c49ad3c898b797c0709/cvae.py

logger = getLogger()


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3, bias: bool = False):
        super(ResBlock, self).__init__()

        p = kernel_size // 2

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=p,
                bias=bias,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=p,
                bias=bias,
            ),
        )
        self.act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.convs(x))


class Conv2dCvae(nn.Module):
    def __init__(
        self,
        *,
        encode_x_feat_channels: int,
        encode_o_feat_channels: int,
        n_encode_blocks: int,
        decode_feat_channels: int,
        n_decode_layers: int,
        bias: bool,
        prior_model: nn.Module,
        **kwargs,
    ):
        super().__init__()

        assert n_encode_blocks > 1
        assert n_decode_layers > 1

        logger.info("This is Conv2dCvae.")

        in_channels = 1
        out_channels = 1
        kernel_size = 3
        p = kernel_size // 2

        self.lr_nx = 32
        self.lr_ny = 16
        self.hr_nx = 128
        self.hr_ny = 64
        self.prior = prior_model

        for param in self.prior.parameters():
            param.requires_grad = False
        self.prior.eval()

        self.encode_x_feat_extractor = nn.Conv2d(
            in_channels,
            encode_x_feat_channels,
            kernel_size=kernel_size,
            padding=p,
            bias=bias,
        )

        self.encode_o_feat_extractor = nn.Sequential(
            EncoderBlock(
                in_channels=in_channels,
                out_channels=encode_o_feat_channels // 2,
                stride=2,
                kernel_size=kernel_size,
                bias=bias,
            ),
            EncoderBlock(
                in_channels=encode_o_feat_channels // 2,
                out_channels=encode_o_feat_channels,
                stride=2,
                kernel_size=kernel_size,
                bias=bias,
            ),
        )

        layers = []
        for i in range(n_encode_blocks):
            layers.append(
                ResBlock(
                    encode_x_feat_channels + encode_o_feat_channels,
                    kernel_size=kernel_size,
                    bias=bias,
                )
            )
        layers.append(
            nn.Conv2d(
                encode_x_feat_channels + encode_o_feat_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=p,
                bias=True,
            )
        )
        self.encoder = nn.Sequential(*layers)

        self.encoder_logvar = nn.Conv2d(
            self.prior.feat_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=p,
            bias=True,
        )

        layers = []
        for i in range(n_decode_layers - 1):
            layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else decode_feat_channels,
                    decode_feat_channels,
                    kernel_size=kernel_size,
                    padding=p,
                    bias=True,
                )
            )
            layers.append(nn.LeakyReLU())
        self.decoder = nn.Sequential(*layers)

        self.decoder_mu = nn.Conv2d(
            decode_feat_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=p,
            bias=True,
        )

    def encode(
        self, x: torch.Tensor, obs: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        # Check channel, x, y dims
        assert x.shape[1:] == (1, self.lr_ny, self.lr_nx)
        assert obs.shape[1:] == (1, self.hr_ny, self.hr_nx)

        fx = self.encode_x_feat_extractor(x)
        fo = self.encode_o_feat_extractor(obs)

        y = torch.cat([fx, fo], dim=1)  # concat along channel dim
        y = self.encoder(y)
        lr = y + x

        y = self.prior.calc_super_resolved_features(lr)

        mu = self.prior.last(y) + self.prior.bicubic(lr)
        logvar = self.encoder_logvar(y)

        return mu, logvar, lr

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:

        # Check channel, x, y dims
        assert z.shape[1:] == (1, self.hr_ny, self.hr_nx)

        y = self.decoder(z)
        y = self.decoder_mu(y)

        return y

    def forward(
        self, x: torch.Tensor, obs: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, lr = self.encode(x, obs)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, lr