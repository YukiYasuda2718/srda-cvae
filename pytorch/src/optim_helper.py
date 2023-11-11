import copy
import datetime
import random
from logging import getLogger

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Sampler

logger = getLogger()


def train_snapshot_cvae_ddp(
    *,
    dataloader: DataLoader,
    sampler: Sampler,
    cvae: nn.Module,
    prior_model: nn.Module,
    vlb_fn: nn.functional,
    optimizer: Optimizer,
    epoch: int,
    rank: int,
    world_size: int,
) -> float:

    mean_loss, cnt = 0.0, 0
    sampler.set_epoch(epoch)
    cvae.train()
    prior_model.eval()

    for x, obs, is_obs, _ in dataloader:
        x, obs, is_obs = x.to(rank), obs.to(rank), is_obs.to(rank)
        obs_reconst, mu, logvar = cvae(x, obs)

        with torch.no_grad():
            mu_prior = prior_model(x)

        vlb = vlb_fn(
            mu=mu,
            logvar=logvar,
            obs_reconst=obs_reconst,
            mu_prior=mu_prior,
            obs=obs,
            is_obs=is_obs,
        )

        optimizer.zero_grad()
        vlb.backward()
        optimizer.step()

        mean_loss += vlb * x.shape[0]
        cnt += x.shape[0]

    mean_loss /= cnt

    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)

    return mean_loss.item() / world_size


def validate_snapshot_cvae_ddp(
    *,
    dataloader: DataLoader,
    sampler: Sampler,
    cvae: nn.Module,
    prior_model: nn.Module,
    vlb_fn: nn.functional,
    epoch: int,
    rank: int,
    world_size: int,
) -> float:

    mean_loss, cnt = 0.0, 0
    sampler.set_epoch(epoch)
    cvae.eval()
    prior_model.eval()

    with torch.no_grad():
        for x, obs, is_obs, _ in dataloader:
            x, obs, is_obs = x.to(rank), obs.to(rank), is_obs.to(rank)

            obs_reconst, mu, logvar = cvae(x, obs)
            mu_prior = prior_model(x)

            vlb = vlb_fn(
                mu=mu,
                logvar=logvar,
                obs_reconst=obs_reconst,
                mu_prior=mu_prior,
                obs=obs,
                is_obs=is_obs,
            )

            mean_loss += vlb * x.shape[0]
            cnt += x.shape[0]

    mean_loss /= cnt

    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)

    return mean_loss.item() / world_size


def train_prior_model_ddp(
    *,
    dataloader: DataLoader,
    sampler: Sampler,
    prior_model: nn.Module,
    loss_fn: nn.functional,
    optimizer: Optimizer,
    epoch: int,
    rank: int,
    world_size: int,
) -> float:

    mean_loss, cnt = 0.0, 0
    sampler.set_epoch(epoch)
    prior_model.train()

    for x, obs, is_obs, _ in dataloader:
        x, obs, is_obs = x.to(rank), obs.to(rank), is_obs.to(rank)
        preds = prior_model(x)

        loss = loss_fn(preds, obs, is_obs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss += loss * x.shape[0]
        cnt += x.shape[0]

    mean_loss /= cnt

    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)

    return mean_loss.item() / world_size


def validate_prior_model_ddp(
    *,
    dataloader: DataLoader,
    sampler: Sampler,
    prior_model: nn.Module,
    loss_fn: nn.functional,
    epoch: int,
    rank: int,
    world_size: int,
) -> float:

    mean_loss, cnt = 0.0, 0
    sampler.set_epoch(epoch)
    prior_model.eval()

    with torch.no_grad():
        for x, obs, is_obs, _ in dataloader:
            x, obs, is_obs = x.to(rank), obs.to(rank), is_obs.to(rank)

            preds = prior_model(x)
            loss = loss_fn(preds, obs, is_obs)

            mean_loss += loss * x.shape[0]
            cnt += x.shape[0]

    mean_loss /= cnt

    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)

    return mean_loss.item() / world_size