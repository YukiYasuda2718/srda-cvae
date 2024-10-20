import typing

import numpy as np
import torch
import torch.nn.functional as F

# This module is a re-implementation from https://github.com/X-Lowlevel-Vision/LAM_Demo (official repository)

# The following methods are to calculate Local Attribution Map (LAM)
# See the article, which proposes LAM
# Jinjin Gu, Chao Dong, 2021, "Interpreting Super-Resolution Networks with Local Attribution Maps"
# https://arxiv.org/abs/2011.11036


def calc_lam(
    data: torch.Tensor,
    model: torch.nn.Module,
    *,
    h: int,
    w: int,
    window: int,
    fold: int,
    l: int,
    sigma: float,
    cuda: bool,
):
    assert data.ndim == 3  # channel, x, and y

    if next(model.parameters()).get_device() >= 0:  # on GPU
        assert cuda == True

    attr_objective = _attribution_objective(_attr_grad, h=h, w=w, window=window)
    gaus_blur_path_func = _gaussian_blur_path(sigma=sigma, fold=fold, l=l)

    if data.get_device() >= 0:  # on GPU
        d = data.cpu()
    else:
        d = data

    path_grad_images, path_sr_images, path_lr_images = _path_gradient(
        image=d,
        model=model,
        attr_objective=attr_objective,
        path_interpolation_func=gaus_blur_path_func,
        cuda=cuda,
    )

    grad_numpy, sr_image = _saliency_map(path_grad_images, path_sr_images)
    lam = _grad_abs_norm(grad_numpy)

    if sr_image.get_device() >= 0:
        sr_image = sr_image.cpu()

    return lam.detach().numpy(), sr_image.detach().numpy()


def _reduce_func(method: str) -> typing.Callable:
    if method == "sum":
        return torch.sum
    elif method == "mean":
        return torch.mean
    elif method == "count":
        return lambda x: sum(x.size())
    else:
        raise NotImplementedError()


def _attr_grad(data: torch.Tensor, h: int, w: int, window: int, reduce: str = "sum"):
    shape = data.shape
    assert len(shape) == 4, f"{shape=}"  # bach, channel, x, and y dims

    d = _pad(data, [1, 1, 1, 1])

    x_grad = d[..., 2:, :] - d[..., :-2, :]
    y_grad = d[..., 2:] - d[..., :-2]
    grad = torch.sqrt(x_grad[..., 1:-1] ** 2 + y_grad[..., 1:-1, :] ** 2)
    assert grad.shape == shape, f"{grad.shape=}"

    crop = grad[..., h : h + window, w : w + window]

    return _reduce_func(reduce)(crop)


def _attribution_objective(
    attr_func: typing.Callable, h: int, w: int, window: int = 16
):
    def calculate_objective(image):
        return attr_func(image, h=h, w=w, window=window)

    return calculate_objective


def _isotropic_gaussian_kernel(l: int, sigma: float, epsilon: float = 1e-5):
    ax = np.arange(-l // 2 + 1.0, l // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * (sigma + epsilon) ** 2))
    return torch.from_numpy(kernel / np.sum(kernel)).to(torch.float32)


def _pad(data: torch.Tensor, pad: typing.List[int]):
    assert data.ndim == 4  # batch, channel, x and y
    assert data.dtype == torch.float32 or data.dtype == torch.float64
    assert len(pad) == 4
    assert pad[0] == pad[1] == pad[2] == pad[3]

    d = F.pad(data, pad=pad, mode="circular")
    _d = F.pad(data, pad=pad, mode="reflect")

    p = pad[0]
    d[..., :p, :] = -_d[..., :p, :]
    d[..., -p:, :] = -_d[..., -p:, :]  # odd-reflection along y dim

    return d


def _zeropad(data: torch.Tensor, pad: typing.List[int]):
    assert data.ndim == 4  # batch, channel, x and y
    assert data.dtype == torch.float32 or data.dtype == torch.float64
    assert len(pad) == 4
    assert pad[0] == pad[1] == pad[2] == pad[3]

    d = F.pad(data, pad=pad, mode="constant", value=0)

    return d


def _filter2d(data: torch.Tensor, kernel: torch.Tensor):
    assert data.ndim == 4  # batch, channel, x, and y
    assert kernel.ndim == 2  # x and y
    assert kernel.shape[0] % 2 == 1
    assert kernel.shape[1] % 2 == 1
    assert kernel.shape[0] == kernel.shape[1]

    in_channels = data.shape[1]
    out_channels = in_channels

    _k = torch.broadcast_to(kernel, (out_channels, 1) + kernel.shape)

    pad = [
        kernel.shape[-1] // 2,
        kernel.shape[-1] // 2,
        kernel.shape[-2] // 2,
        kernel.shape[-2] // 2,
    ]

    d = _zeropad(data, pad)

    return F.conv2d(d, _k, groups=in_channels)


def _gaussian_blur_path(sigma: float, fold: int, l: int):
    def path_interpolation_func(image: torch.Tensor):
        assert image.ndim == 3  # channel, x, and y

        sigma_interpolation = np.linspace(sigma, 0, fold + 1)
        kernel_interpolation = torch.zeros((fold + 1, l, l))
        for i in range(fold + 1):
            kernel_interpolation[i] = _isotropic_gaussian_kernel(
                l=l, sigma=sigma_interpolation[i]
            )

        image_interpolation = torch.zeros((fold,) + image.shape)
        lambda_derivative_interpolation = torch.zeros((fold,) + image.shape)

        for i in range(fold):
            image_interpolation[i] = _filter2d(
                image[None, ...], kernel_interpolation[i + 1]
            ).squeeze(0)

            lambda_derivative_interpolation[i] = _filter2d(
                image[None, ...],
                (kernel_interpolation[i + 1] - kernel_interpolation[i]) * fold,
            ).squeeze(0)
        return image_interpolation, lambda_derivative_interpolation

    return path_interpolation_func


def _path_gradient(
    image: torch.Tensor,
    model: torch.nn.Module,
    attr_objective: typing.Callable,
    path_interpolation_func: typing.Callable,
    cuda: bool = False,
):

    assert image.ndim == 3  # channel, x, and y

    if cuda:
        model = model.cuda()

    image_interpolation, derivative_interpolation = path_interpolation_func(image)

    grad_accumulate_list = torch.zeros_like(image_interpolation)
    result_list = []

    for i in range(image_interpolation.shape[0]):
        img_tensor = image_interpolation[i].detach().clone()
        assert img_tensor.ndim == 3  # channel, x and y
        img_tensor.requires_grad_(True)

        if cuda:
            result = model(img_tensor[None, ...].cuda())
            target = attr_objective(result)
            target.backward()
            grad = img_tensor.grad.cpu()
            if torch.any(torch.isnan(grad)).item():
                grad[torch.isnan(grad)] = 0.0
        else:
            result = model(img_tensor[None, ...])
            target = attr_objective(result)
            target.backward()
            grad = img_tensor.grad
            if torch.any(torch.isnan(grad)).item():
                grad[torch.isnan(grad)] = 0.0

        deri = derivative_interpolation[i]
        assert grad.shape == deri.shape, f"{grad.shape=}, {deri.shape=}"

        grad_accumulate_list[i] = grad * deri
        result_list.append(result.squeeze(0))

    results = torch.stack(result_list, dim=0)

    return grad_accumulate_list, results, image_interpolation


def _saliency_map(grad_list: torch.Tensor, result_list: torch.Tensor):
    final_grad = torch.mean(grad_list, dim=0)
    return final_grad, result_list[-1]


def _grad_abs_norm(grad):
    assert grad.ndim == 3  # channel, x, and y
    grad_2d = torch.abs(torch.sum(grad, dim=0))
    grad_max = torch.max(grad)
    assert grad_max.ndim == 0
    return grad_2d / grad_max