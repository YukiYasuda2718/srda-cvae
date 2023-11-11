import os
import typing
from logging import getLogger

import numpy as np
import torch
from cfd_model.cfd.periodic_channel_domain import TorchSpectralModel2D
from cfd_model.initialization.periodic_channel_jet_initializer import calc_jet_forcing
from ml_model.conv2d_sr_v2 import ConvSrNetVer02
from src.dataset import DatasetVorticitySnapshot
from src.model_maker import make_model
from src.utils import read_pickle

LR_NX = 32
LR_NY = 17
LR_DT = 5e-4
LR_NT = 500

HR_NX = 128
HR_NY = 65
HR_DT = LR_DT / 4.0
HR_NT = LR_NT * 4

assert HR_DT * HR_NT == LR_DT * LR_NT

N_CYCLES = 96

CFD_BETA = 0.1
COEFF_LINEAR_DRAG = 1e-2
ORDER_DIFFUSION = 2
HR_COEFF_DIFFUSION = 1e-5
LR_COEFF_DIFFUSION = 5e-5

DT = LR_DT * LR_NT

Y0_MEAN = np.pi / 2.0
SIGMA_MEAN = 0.4
U0_MEAN = 3.0
TAU0_MEAN = 0.3

logger = getLogger()


def make_prior_model(weight_path: str, device: str):
    prior_model = ConvSrNetVer02()
    prior_model = prior_model.to(device)
    _ = prior_model.load_state_dict(torch.load(weight_path, map_location=device))
    _ = prior_model.eval()
    return prior_model


def make_cvae(weight_path: str, config: dict, device: str):
    cvae = make_model(config)
    cvae = cvae.to(device)
    _ = cvae.load_state_dict(torch.load(weight_path, map_location=device))
    _ = cvae.eval()
    return cvae


def read_hr_omega_and_obsrv(dir_path: str, dtype: torch.dtype = torch.float64):
    # result_01234 --> 01234
    i_sim = os.path.basename(dir_path).split("_")[-1]
    hr_omega = torch.from_numpy(np.load(f"{dir_path}/hr_omegas_{i_sim}.npy")).to(dtype)
    hr_obsrv = torch.from_numpy(np.load(f"{dir_path}/hr_obsrvs_{i_sim}.npy")).to(dtype)
    model_params = read_pickle(f"{dir_path}/model_params_{i_sim}.pickle")
    return hr_omega, hr_obsrv, model_params


def read_hr_omegas_and_obsrvs(
    dir_paths: typing.List[str], dtype: torch.dtype = torch.float64
) -> torch.Tensor:

    hr_omegas, hr_obsrvs, model_params = [], [], []
    for dir_path in dir_paths:
        hr_omega, hr_obsrv, _params = read_hr_omega_and_obsrv(dir_path, dtype)
        hr_omegas.append(hr_omega)
        hr_obsrvs.append(hr_obsrv)
        model_params.append(_params)
    return torch.cat(hr_omegas, dim=0), torch.cat(hr_obsrvs, dim=0), model_params


def make_cfd_models(device: str):
    lr_model = TorchSpectralModel2D(
        nx=LR_NX,
        ny=LR_NY,
        coeff_linear_drag=COEFF_LINEAR_DRAG,
        coeff_diffusion=LR_COEFF_DIFFUSION,
        order_diffusion=ORDER_DIFFUSION,
        beta=CFD_BETA,
        device=device,
    )

    srda_model = TorchSpectralModel2D(
        nx=LR_NX,
        ny=LR_NY,
        coeff_linear_drag=COEFF_LINEAR_DRAG,
        coeff_diffusion=LR_COEFF_DIFFUSION,
        order_diffusion=ORDER_DIFFUSION,
        beta=CFD_BETA,
        device=device,
    )

    return lr_model, srda_model


def make_jet_forcing(n_ensemble: int):
    _, lr_forcing = calc_jet_forcing(
        nx=LR_NX,
        ny=LR_NY,
        ne=1,
        y0=Y0_MEAN,
        sigma=SIGMA_MEAN,
        tau0=TAU0_MEAN,
    )
    return torch.broadcast_to(lr_forcing, (n_ensemble, LR_NX, LR_NY))


def preprocess(data: torch.Tensor, dataset: DatasetVorticitySnapshot) -> torch.Tensor:
    clamp_max = dataset.clamp_max
    clamp_min = dataset.clamp_min
    bias = dataset.bias
    scale = dataset.scale
    missing_value = dataset.missing_value

    logger.debug(f"clamp: min = {clamp_min}, max = {clamp_max}")
    logger.debug(f"bias = {bias}, scale = {scale}")
    logger.debug(f"missing value = {missing_value}")

    # drop the last index of y
    data = data[..., :-1].to(torch.float32)
    data = (data - bias) / scale
    data = torch.clamp(data, min=clamp_min, max=clamp_max)

    data = torch.nan_to_num(data, nan=missing_value)

    return data


def append_zeros(data: torch.Tensor):
    assert data.ndim == 4  # ensemble, channel, x, and y

    zs = torch.zeros(data.shape[0:3], dtype=data.dtype, device=data.device)[..., None]

    # Concat along y dim
    appended = torch.cat([data, zs], dim=-1)

    # Check the last index of y has zero values
    assert torch.max(torch.abs(appended[..., -1])).item() == 0.0

    return appended


def inv_preprocess(
    data: torch.Tensor, dataset: DatasetVorticitySnapshot
) -> torch.Tensor:

    bias = dataset.bias
    scale = dataset.scale

    logger.debug(f"bias = {bias}, scale = {scale}")

    data = data * scale + bias

    data = append_zeros(data)

    return data.to(torch.float64)


def add_system_noise(sys_noise_generator, n_ensemble, lr_model, device):
    noise = sys_noise_generator.sample([n_ensemble]).reshape(n_ensemble, LR_NX, LR_NY)
    noise = noise - torch.mean(noise, dim=0, keepdims=True)
    omega0 = lr_model.omega + noise.to(device)
    lr_model.initialize(t0=lr_model.t, omega0=omega0)
    lr_model.calc_grid_data()