import typing
from logging import getLogger

import torch
import torch.nn as nn

logger = getLogger()


class EncoderBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel_size: int = 3,
        bias: bool = False,
        type_down_sample: typing.Literal["conv", "average"] = "conv",
        num_layers: int = 2,
        negative_slope: float = 0.0,
    ):
        super(EncoderBlock, self).__init__()

        assert num_layers >= 2

        p = kernel_size // 2

        if type_down_sample == "conv":
            self.down = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=p,
                    bias=bias,
                ),
                nn.LeakyReLU(negative_slope=negative_slope),
            )
        elif type_down_sample == "average":
            self.down = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=p,
                    bias=bias,
                ),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0),
            )
        else:
            raise Exception(f"Downsampling type of {type_down_sample} is not supported")

        convs = []
        for _ in range(num_layers - 1):
            convs.append(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=p,
                    bias=bias,
                )
            )
            convs.append(nn.LeakyReLU(negative_slope=negative_slope))

        self.convs = nn.Sequential(*convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(x)
        return self.convs(y)


def ICNR(
    tensor: torch.Tensor, scale_factor: int = 2, initializer=nn.init.kaiming_uniform_
):
    OUT, IN, H, W = tensor.shape
    sub = torch.zeros(OUT // scale_factor**2, IN, H, W)
    sub = initializer(sub)

    kernel = torch.zeros_like(tensor)
    for i in range(OUT):
        kernel[i] = sub[i // scale_factor**2]

    return kernel


class PixelShuffleBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        kernel_size: int = 3,
        bias: bool = False,
        negative_slope: float = 1e-2,
        upscale: int = 2,
        initialization: str = None,
    ):
        super(PixelShuffleBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            (upscale**2) * in_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )

        if initialization is not None and initialization == "ICNR":
            logger.info("ICNR initialization is used in PixelShuffleBlock.")
            kernel = ICNR(self.conv.weight, upscale)
            self.conv.weight.data.copy_(kernel)

        self.act = nn.LeakyReLU(negative_slope=negative_slope)

        self.upsample = nn.PixelShuffle(upscale_factor=upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.act(y)
        return self.upsample(y)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = False,
        type_up_sample: typing.Literal["pixel_shuffle"] = "pixel_shuffle",
        num_layers: int = 2,
        negative_slope: float = 1e-2,
        upscale: int = 2,
        initialization: str = None,
    ):
        super(DecoderBlock, self).__init__()

        assert num_layers >= 2

        if type_up_sample == "pixel_shuffle":
            self.up = PixelShuffleBlock(
                in_channels=in_channels,
                kernel_size=kernel_size,
                bias=bias,
                negative_slope=negative_slope,
                upscale=upscale,
                initialization=initialization,
            )
        else:
            raise Exception(f"Upsampling type of {type_up_sample} is not supported")

        convs = []
        for i in range(num_layers - 1):
            convs.append(
                nn.Conv2d(
                    (in_channels if i == 0 else out_channels),
                    out_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                    bias=bias,
                )
            )
            convs.append(nn.LeakyReLU(negative_slope=negative_slope))

        self.convs = nn.Sequential(*convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.up(x)
        return self.convs(y)