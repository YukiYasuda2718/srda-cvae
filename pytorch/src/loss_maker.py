from logging import getLogger

import numpy as np
import torch
from torch import nn

logger = getLogger()


def make_loss(config: dict) -> nn.Module:

    if config["train"]["loss"]["name"] == "L1":
        logger.info("L1 loss is created.")
        return nn.L1Loss(reduction="mean")

    elif config["train"]["loss"]["name"] == "MSE":
        logger.info("MSE loss is created.")
        return nn.MSELoss(reduction="mean")

    else:
        raise NotImplementedError(
            f'{config["train"]["loss"]["name"]} is not supported.'
        )


class MaskedL1Loss(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        abs_diff = torch.abs(predicts - targets)

        return torch.sum(masks * abs_diff) / (torch.sum(masks) + self.eps)


class DummyL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        abs_diff = torch.abs(predicts - targets)

        return torch.mean(abs_diff)


class VariationalLowerBoundSnapshot(nn.Module):
    def __init__(self, std_latent: float, std_reconst: float, beta: float, **kwargs):
        super().__init__()
        self.std_latent = std_latent
        self.std_reconst = std_reconst
        self.beta = beta

        logger.info(f"std_latent = {self.std_latent}")
        logger.info(f"std_reconst = {self.std_reconst}")
        logger.info(f"beta = {self.beta}")

    def forward(
        self,
        *,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        obs_reconst: torch.Tensor,
        mu_prior: torch.Tensor,
        obs: torch.Tensor,
        is_obs: torch.Tensor,
    ) -> torch.Tensor:

        kl_div = 2 * np.log(self.std_latent) - torch.mean(logvar)
        kl_div = kl_div + torch.mean(torch.exp(logvar)) / (self.std_latent**2)
        kl_div = kl_div + torch.mean((mu - mu_prior) ** 2) / (self.std_latent**2)

        negative_log_likelihood = torch.mean(is_obs * ((obs - obs_reconst) ** 2))
        negative_log_likelihood = negative_log_likelihood / (self.std_reconst**2)

        return self.beta * kl_div + negative_log_likelihood


class VariationalLowerBoundSnapshotWithoutMask(nn.Module):
    def __init__(self, std_latent: float, std_reconst: float, beta: float, **kwargs):
        super().__init__()
        self.std_latent = std_latent
        self.std_reconst = std_reconst
        self.beta = beta

        logger.info(f"std_latent = {self.std_latent}")
        logger.info(f"std_reconst = {self.std_reconst}")
        logger.info(f"beta = {self.beta}")

    def forward(
        self,
        *,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        obs_reconst: torch.Tensor,
        mu_prior: torch.Tensor,
        obs: torch.Tensor,
    ) -> torch.Tensor:

        kl_div = 2 * np.log(self.std_latent) - torch.mean(logvar)
        kl_div = kl_div + torch.mean(torch.exp(logvar)) / (self.std_latent**2)
        kl_div = kl_div + torch.mean((mu - mu_prior) ** 2) / (self.std_latent**2)

        negative_log_likelihood = torch.mean((obs - obs_reconst) ** 2)
        negative_log_likelihood = negative_log_likelihood / (self.std_reconst**2)

        return self.beta * kl_div + negative_log_likelihood