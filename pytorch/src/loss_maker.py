from logging import getLogger

import numpy as np
import torch
from torch import nn

logger = getLogger()


class MaskedL1Loss(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        abs_diff = torch.abs(predicts - targets)

        return torch.sum(masks * abs_diff) / (torch.sum(masks) + self.eps)


class VariationalLowerBoundSnapshot(nn.Module):
    def __init__(
        self,
        std_latent: float,
        std_reconst: float,
        beta: float,
        lr_weight: float,
        **kwargs,
    ):
        super().__init__()
        self.std_latent = std_latent
        self.std_reconst = std_reconst
        self.beta = beta
        self.lr_weight = lr_weight

        logger.info(f"std_latent = {self.std_latent}")
        logger.info(f"std_reconst = {self.std_reconst}")
        logger.info(f"beta = {self.beta}")
        logger.info(f"lr_weight = {self.lr_weight}")

    def forward(
        self,
        *,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        obs_reconst: torch.Tensor,
        mu_prior: torch.Tensor,
        obs: torch.Tensor,
        is_obs: torch.Tensor,
        lr_analysis: torch.Tensor,
        lr_input: torch.Tensor,
    ) -> torch.Tensor:

        kl_div = 2 * np.log(self.std_latent) - torch.mean(logvar)
        kl_div = kl_div + torch.mean(torch.exp(logvar)) / (self.std_latent**2)
        kl_div = kl_div + torch.mean((mu - mu_prior) ** 2) / (self.std_latent**2)

        lr_los = torch.mean((lr_analysis - lr_input) ** 2)

        negative_log_likelihood = torch.mean(is_obs * ((obs - obs_reconst) ** 2))
        negative_log_likelihood = negative_log_likelihood / (self.std_reconst**2)

        return self.beta * kl_div + negative_log_likelihood + self.lr_weight * lr_los