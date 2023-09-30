from logging import getLogger

import torch

logger = getLogger()


def append_zeros(data: torch.Tensor):
    assert data.ndim == 4  # ens, time, x, and y

    zs = torch.zeros(data.shape[0:3], dtype=data.dtype)[..., None]

    appended = torch.cat([data, zs], dim=-1)

    # Check the last index of y has zero values
    assert torch.max(torch.abs(appended[..., -1])).item() == 0.0

    return appended


def _preprocess(
    data: torch.Tensor,
    biases: torch.Tensor,
    scales: torch.Tensor,
    clamp_min: float,
    clamp_max: float,
):
    ret = (data - biases) / scales
    return torch.clamp(ret, min=clamp_min, max=clamp_max)


def preprocess(
    *,
    data: torch.Tensor,
    biases: torch.Tensor,
    scales: torch.Tensor,
    clamp_min: float,
    clamp_max: float,
    n_ens: int,
    assimilation_period: int,
    ny: int,
    nx: int,
    device: str,
    dtype: torch.dtype = torch.float32,
):
    # time, ens, x, y --> ens, time, y, x
    data = data[..., :-1].permute(1, 0, 3, 2).contiguous()
    data = data.unsqueeze(2)  # Add channel dim
    assert data.shape[0] == n_ens
    assert data.ndim == 5  # batch, time, channel, y, x

    data = _preprocess(data, biases, scales, clamp_min, clamp_max)
    data = data.to(dtype).to(device)
    assert data.shape[0] == n_ens
    assert data.ndim == 5  # batch, time, channel, y, x

    return data


def inv_preprocess(data: torch.Tensor, biases: torch.Tensor, scales: torch.Tensor):
    return data * scales + biases