import copy
import datetime
import random
import sys
import time
import typing
from logging import getLogger

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from src.early_stopping import EarlyStopping
from src.loss_maker import MaskedL1Loss, VariationalLowerBoundSnapshot
from src.utils import AverageMeter, froze_model_params, unfroze_model_params
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Sampler

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


logger = getLogger()


def train(
    *,
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.functional,
    optimizer: Optimizer,
    device: str,
    hide_progress_bar: bool = True,
) -> float:

    train_loss = AverageMeter()
    model.train()

    with tqdm(total=len(dataloader), disable=hide_progress_bar) as t:
        for Xs, obs, ys in dataloader:
            Xs, obs, ys = Xs.to(device), obs.to(device), ys.to(device)

            preds = model(Xs, obs)
            loss = loss_fn(preds, ys)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), n=len(Xs))
            t.update(1)

    logger.info(f"Train error: avg loss = {train_loss.avg:.8f}")

    return train_loss.avg


def train_prior_model(
    *,
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.functional,
    optimizer: Optimizer,
    device: str,
    hide_progress_bar: bool = True,
) -> float:

    train_loss = AverageMeter()
    model.train()

    with tqdm(total=len(dataloader), disable=hide_progress_bar) as t:
        for Xs, obs, is_obs, _ in dataloader:
            Xs, obs, is_obs = Xs.to(device), obs.to(device), is_obs.to(device)

            preds = model(Xs)
            loss = loss_fn(preds, obs, is_obs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), n=len(Xs))
            t.update(1)

    logger.info(f"Train error: avg loss = {train_loss.avg:.8f}")

    return train_loss.avg


def train_prior_model_using_ground_truth(
    *,
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.functional,
    optimizer: Optimizer,
    device: str,
    hide_progress_bar: bool = True,
) -> float:

    train_loss = AverageMeter()
    model.train()

    with tqdm(total=len(dataloader), disable=hide_progress_bar) as t:
        for Xs, _, is_obs, gt in dataloader:
            Xs, is_obs, gt = Xs.to(device), is_obs.to(device), gt.to(device)

            preds = model(Xs)
            loss = loss_fn(preds, gt, is_obs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), n=len(Xs))
            t.update(1)

    logger.info(f"Train error: avg loss = {train_loss.avg:.8f}")

    return train_loss.avg


def validate_prior_model(
    *,
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.functional,
    device: str,
    hide_progress_bar: bool = True,
) -> float:

    val_loss = AverageMeter()
    model.eval()

    with tqdm(total=len(dataloader), disable=hide_progress_bar) as t:
        with torch.no_grad():
            for Xs, obs, is_obs, _ in dataloader:
                Xs, obs, is_obs = Xs.to(device), obs.to(device), is_obs.to(device)

                preds = model(Xs)
                val_loss.update(loss_fn(preds, obs, is_obs).item(), n=len(Xs))
                t.update(1)

    logger.info(f"Valid error: avg loss = {val_loss.avg:.8f}")

    return val_loss.avg


def validate_prior_model_using_ground_truth(
    *,
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.functional,
    device: str,
    hide_progress_bar: bool = True,
) -> float:

    val_loss = AverageMeter()
    model.eval()

    with tqdm(total=len(dataloader), disable=hide_progress_bar) as t:
        with torch.no_grad():
            for Xs, _, is_obs, gt in dataloader:
                Xs, is_obs, gt = Xs.to(device), is_obs.to(device), gt.to(device)

                preds = model(Xs)
                val_loss.update(loss_fn(preds, gt, is_obs).item(), n=len(Xs))
                t.update(1)

    logger.info(f"Valid error: avg loss = {val_loss.avg:.8f}")

    return val_loss.avg


def train_snapshot_cvae(
    *,
    dataloader: DataLoader,
    cvae: nn.Module,
    prior_model: nn.Module,
    vlb_fn: nn.functional,
    optimizer: Optimizer,
    device: str,
    hide_progress_bar: bool = True,
) -> float:

    train_loss = AverageMeter()
    cvae.train()
    prior_model.eval()

    with tqdm(total=len(dataloader), disable=hide_progress_bar) as t:
        for x, obs, is_obs, _ in dataloader:
            x, obs, is_obs = x.to(device), obs.to(device), is_obs.to(device)
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

            train_loss.update(vlb.item(), n=len(x))
            t.update(1)

    logger.info(f"Train error: avg loss = {train_loss.avg:.8f}")

    return train_loss.avg


def validate_snapshot_cvae(
    *,
    dataloader: DataLoader,
    cvae: nn.Module,
    prior_model: nn.Module,
    vlb_fn: nn.functional,
    device: str,
    hide_progress_bar: bool = True,
) -> float:

    valid_loss = AverageMeter()
    cvae.eval()
    prior_model.eval()

    with tqdm(total=len(dataloader), disable=hide_progress_bar) as t:
        with torch.no_grad():
            for x, obs, is_obs, _ in dataloader:
                x, obs, is_obs = x.to(device), obs.to(device), is_obs.to(device)

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

                valid_loss.update(vlb.item(), n=len(x))
                t.update(1)

    logger.info(f"Validation error: avg loss = {valid_loss.avg:.8f}")

    return valid_loss.avg


def test(
    *,
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.functional,
    device: str,
    hide_progress_bar: bool = True,
) -> float:

    val_loss = AverageMeter()
    model.eval()

    with tqdm(total=len(dataloader), disable=hide_progress_bar) as t:
        with torch.no_grad():
            for Xs, obs, ys in dataloader:
                Xs, obs, ys = Xs.to(device), obs.to(device), ys.to(device)

                preds = model(Xs, obs)
                val_loss.update(loss_fn(preds, ys).item(), n=len(Xs))
                t.update(1)

    logger.info(f"Valid error: avg loss = {val_loss.avg:.8f}")

    return val_loss.avg


def test_snapshot_model(
    *,
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.functional,
    device: str,
    hide_progress_bar: bool = True,
) -> float:

    val_loss = AverageMeter()
    model.eval()

    with tqdm(total=len(dataloader), disable=hide_progress_bar) as t:
        with torch.no_grad():
            for Xs, _, _, ys in dataloader:
                Xs, ys = Xs.to(device), ys.to(device)

                preds = model(Xs)
                val_loss.update(loss_fn(preds, ys).item(), n=len(Xs))
                t.update(1)

    logger.info(f"Test error: avg loss = {val_loss.avg:.8f}")

    return val_loss.avg


def test_snapshot_cvae(
    *,
    dataloader: DataLoader,
    cvae: nn.Module,
    loss_fn: nn.functional,
    device: str,
    hide_progress_bar: bool = True,
) -> float:

    valid_loss = AverageMeter()
    cvae.eval()

    with tqdm(total=len(dataloader), disable=hide_progress_bar) as t:
        with torch.no_grad():
            for x, obs, _, y in dataloader:
                x, obs, y = x.to(device), obs.to(device), y.to(device)

                _, mu, _ = cvae(x, obs)

                valid_loss.update(loss_fn(mu, y).item(), n=len(x))
                t.update(1)

    logger.info(f"Test error: avg loss = {valid_loss.avg:.8f}")

    return valid_loss.avg


def train_ddp(
    *,
    dataloader: DataLoader,
    sampler: Sampler,
    model: nn.Module,
    loss_fn: nn.functional,
    optimizer: Optimizer,
    epoch: int,
    rank: int,
    world_size: int,
) -> float:

    mean_loss, cnt = 0.0, 0

    random.seed(epoch)
    np.random.seed(epoch)
    sampler.set_epoch(epoch)
    model.train()

    for Xs, obs, ys in dataloader:
        Xs, obs, ys = Xs.to(rank), obs.to(rank), ys.to(rank)

        preds = model(Xs, obs)
        loss = loss_fn(preds, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss += loss * Xs.shape[0]
        cnt += Xs.shape[0]
    mean_loss /= cnt

    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)

    return mean_loss.item() / world_size


def test_ddp(
    *,
    dataloader: DataLoader,
    sampler: Sampler,
    model: nn.Module,
    loss_fn: nn.functional,
    epoch: int,
    rank: int,
    world_size: int,
) -> float:

    mean_loss, cnt = 0.0, 0

    random.seed(epoch)
    np.random.seed(epoch)
    sampler.set_epoch(epoch)
    model.eval()

    with torch.no_grad():
        for Xs, obs, ys in dataloader:
            Xs, obs, ys = Xs.to(rank), obs.to(rank), ys.to(rank)

            preds = model(Xs, obs)
            loss = loss_fn(preds, ys)

            mean_loss += loss * Xs.shape[0]
            cnt += Xs.shape[0]
    mean_loss /= cnt

    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)

    return mean_loss.item() / world_size


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


def unspervised_train_vae(
    *,
    dataloader: DataLoader,
    vae: nn.Module,
    vlb_fn: nn.functional,
    optimizer: Optimizer,
    device: str,
    epoch: int,
    hide_progress_bar: bool = True,
) -> float:

    random.seed(epoch)
    np.random.seed(epoch)

    train_loss = AverageMeter()
    vae.train()

    with tqdm(total=len(dataloader), disable=hide_progress_bar) as t:
        for x, _, _, obs, is_obs in dataloader:

            x, obs, is_obs = x.to(device), obs.to(device), is_obs.to(device)

            obs_reconst, mu, logvar = vae(x, obs)

            vlb = vlb_fn(
                mu=mu,
                logvar=logvar,
                obs_reconst=obs_reconst,
                mu_prior=x,
                obs=obs,
                is_obs=is_obs,
            )

            optimizer.zero_grad()
            vlb.backward()
            optimizer.step()

            train_loss.update(vlb.item(), n=len(x))
            t.update(1)

    logger.info(f"Train error: avg loss = {train_loss.avg:.8f}")

    return train_loss.avg


def unspervised_validate_vae(
    *,
    dataloader: DataLoader,
    vae: nn.Module,
    vlb_fn: nn.functional,
    device: str,
    epoch: int,
    hide_progress_bar: bool = True,
) -> float:

    random.seed(epoch)
    np.random.seed(epoch)

    valid_loss = AverageMeter()
    vae.eval()

    with tqdm(total=len(dataloader), disable=hide_progress_bar) as t:
        with torch.no_grad():
            for x, _, _, obs, is_obs in dataloader:

                x, obs, is_obs = x.to(device), obs.to(device), is_obs.to(device)

                obs_reconst, mu, logvar = vae(x, obs)

                vlb = vlb_fn(
                    mu=mu,
                    logvar=logvar,
                    obs_reconst=obs_reconst,
                    mu_prior=x,
                    obs=obs,
                    is_obs=is_obs,
                )

                valid_loss.update(vlb.item(), n=len(x))
                t.update(1)

    logger.info(f"Valid error: avg loss = {valid_loss.avg:.8f}")

    return valid_loss.avg


def unspervised_train_vae_ddp(
    *,
    dataloader: DataLoader,
    sampler: Sampler,
    vae: nn.Module,
    vlb_fn: nn.functional,
    optimizer: Optimizer,
    epoch: int,
    rank: int,
    world_size: int,
) -> float:

    mean_loss, cnt = 0.0, 0

    sampler.set_epoch(epoch)
    random.seed(epoch)
    np.random.seed(epoch)

    vae.train()

    for x, _, _, obs, is_obs in dataloader:
        x, obs, is_obs = x.to(rank), obs.to(rank), is_obs.to(rank)

        obs_reconst, mu, logvar = vae(x, obs)

        vlb = vlb_fn(
            mu=mu,
            logvar=logvar,
            obs_reconst=obs_reconst,
            mu_prior=x,
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


def unspervised_validate_vae_ddp(
    *,
    dataloader: DataLoader,
    sampler: Sampler,
    vae: nn.Module,
    vlb_fn: nn.functional,
    epoch: int,
    rank: int,
    world_size: int,
) -> float:

    mean_loss, cnt = 0.0, 0

    sampler.set_epoch(epoch)
    random.seed(epoch)
    np.random.seed(epoch)

    vae.eval()

    with torch.no_grad():
        for x, _, _, obs, is_obs in dataloader:
            x, obs, is_obs = x.to(rank), obs.to(rank), is_obs.to(rank)

            obs_reconst, mu, logvar = vae(x, obs)

            vlb = vlb_fn(
                mu=mu,
                logvar=logvar,
                obs_reconst=obs_reconst,
                mu_prior=x,
                obs=obs,
                is_obs=is_obs,
            )

            mean_loss += vlb * x.shape[0]
            cnt += x.shape[0]

    mean_loss /= cnt

    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)

    return mean_loss.item() / world_size


def _train_vae_prior(
    *,
    dataloader: DataLoader,
    prior: nn.Module,
    loss_fn: nn.functional,
    optimizer: Optimizer,
    device: str,
    epoch: int,
    mode: typing.Literal["train", "valid"],
    hide_progress_bar: bool = True,
) -> float:

    random.seed(epoch)
    np.random.seed(epoch)

    loss_meter = AverageMeter()

    if mode == "train":
        prior.train()
    elif mode == "valid":
        prior.eval()
    else:
        raise Exception(f"{mode} is not supported.")

    with tqdm(total=len(dataloader), disable=hide_progress_bar) as t:
        for _, x, _, obs, is_obs in dataloader:

            x, obs, is_obs = x.to(device), obs.to(device), is_obs.to(device)

            if mode == "train":
                preds = prior(x)
                loss = loss_fn(predicts=preds, targets=obs, masks=is_obs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    preds = prior(x)
                    loss = loss_fn(predicts=preds, targets=obs, masks=is_obs)

            loss_meter.update(loss.item(), n=len(x))
            t.update(1)

    logger.info(f"{mode} error: avg loss = {loss_meter.avg:.8f}")

    return loss_meter.avg


def train_vae_prior(
    *,
    prior: nn.Module,
    dict_dataloaders: dict,
    config: dict,
    weight_path: str,
    loss_csv_path: str,
    device: str,
):
    assert config["pretrain"]["loss"]["name"] == "MaskedL1Loss"
    loss_fn = MaskedL1Loss()
    logger.info("Loss: MaskedL1Loss")

    prior = prior.to(device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=config["pretrain"]["lr"])
    logger.info(f'Learning rate = {config["pretrain"]["lr"]}')

    esp = config["pretrain"]["early_stopping_patience"]
    early_stopping = EarlyStopping(early_stopping_patience=esp, logger=logger)

    all_scores = []
    best_epoch = 0
    best_loss = np.inf
    best_weights = copy.deepcopy(prior.state_dict())

    start_time = time.time()
    logger.info(f"\nTrain start: {datetime.datetime.utcnow().isoformat()} UTC")

    for epoch in range(config["pretrain"]["epochs"]):
        _time = time.time()
        logger.info(f'Epoch: {epoch + 1} / {config["pretrain"]["epochs"]}')

        losses = {}
        for mode in ["train", "valid"]:
            loss = _train_vae_prior(
                dataloader=dict_dataloaders[mode],
                prior=prior,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                mode=mode,
            )
            losses[mode] = loss

        all_scores.append(losses)

        if early_stopping(losses["valid"]):
            logger.info("Training is finished.")
            logger.info("-----")
            break

        if losses["valid"] <= best_loss:
            best_epoch = epoch + 1
            best_loss = losses["valid"]
            logger.info("Best loss is updated.")

            best_weights = copy.deepcopy(prior.state_dict())
            torch.save(best_weights, weight_path)

        if epoch % 10 == 0:
            pd.DataFrame(all_scores).to_csv(loss_csv_path, index=False)

        logger.info(f"Elapsed time = {time.time() - _time} sec")
        logger.info("-----")

    pd.DataFrame(all_scores).to_csv(loss_csv_path, index=False)
    end_time = time.time()

    logger.info(f"Best epoch: {best_epoch}, best_loss: {best_loss:.8f}")
    logger.info(f"Train end: {datetime.datetime.utcnow().isoformat()} UTC")
    logger.info(f"Total elapsed time = {(end_time - start_time) / 60.} min")


def _train_vae_encoder_decoder(
    *,
    dataloader: DataLoader,
    encoder: nn.Module,
    decoder: nn.Module,
    prior: nn.Module,
    loss_fn: nn.functional,
    optimizer: Optimizer,
    device: str,
    epoch: int,
    mode: typing.Literal["train", "valid"],
    hide_progress_bar: bool = True,
) -> float:

    random.seed(epoch)
    np.random.seed(epoch)

    loss_meter = AverageMeter()

    prior.eval()

    if mode == "train":
        encoder.train()
        decoder.train()
    elif mode == "valid":
        encoder.eval()
        decoder.eval()
    else:
        raise Exception(f"{mode} is not supported.")

    with tqdm(total=len(dataloader), disable=hide_progress_bar) as t:
        for x, _, _, obs, is_obs in dataloader:

            x, obs, is_obs = x.to(device), obs.to(device), is_obs.to(device)

            with torch.no_grad():
                mu_prior = prior(x)

            if mode == "train":
                mu, logvar = encoder(x=x, obs=obs)
                z = encoder.reparameterize(mu=mu, logvar=logvar)
                obs_reconst = decoder(z)

                loss = loss_fn(
                    mu=mu,
                    logvar=logvar,
                    obs_reconst=obs_reconst,
                    mu_prior=mu_prior,
                    obs=obs,
                    is_obs=is_obs,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    mu, logvar = encoder(x=x, obs=obs)
                    z = encoder.reparameterize(mu=mu, logvar=logvar)
                    obs_reconst = decoder(z)

                    loss = loss_fn(
                        mu=mu,
                        logvar=logvar,
                        obs_reconst=obs_reconst,
                        mu_prior=mu_prior,
                        obs=obs,
                        is_obs=is_obs,
                    )

            loss_meter.update(loss.item(), n=len(x))
            t.update(1)

    logger.info(f"{mode} error: avg loss = {loss_meter.avg:.8f}")

    return loss_meter.avg


def train_vae_encoder_decoder(
    *,
    dict_dataloaders: dict,
    encoder: nn.Module,
    decoder: nn.Module,
    prior: nn.Module,
    config: dict,
    loss_csv_path: str,
    encoder_weight_path: str,
    decoder_weight_path: str,
    device: str,
):

    assert config["train"]["loss"]["name"] == "VariationalLowerBoundSnapshot"
    logger.info("Loss: VariationalLowerBoundSnapshot")
    loss_fn = VariationalLowerBoundSnapshot(**config["train"]["loss"])

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    prior = prior.to(device)

    froze_model_params(prior)

    optimizer = torch.optim.Adam(
        [
            {"params": encoder.parameters(), "lr": config["train"]["encoder_lr"]},
            {"params": decoder.parameters(), "lr": config["train"]["decoder_lr"]},
        ]
    )
    logger.info(f'encoder lr : {config["train"]["encoder_lr"]:.3e}')
    logger.info(f'decoder lr : {config["train"]["decoder_lr"]:.3e}')

    esp = config["train"]["early_stopping_patience"]
    early_stopping = EarlyStopping(early_stopping_patience=esp, logger=logger)

    all_scores = []
    best_epoch = 0
    best_loss = np.inf

    start_time = time.time()
    logger.info(f"\nTrain start: {datetime.datetime.utcnow().isoformat()} UTC")

    for epoch in range(config["train"]["epochs"]):
        _time = time.time()
        logger.info(f'Epoch: {epoch + 1} / {config["train"]["epochs"]}')

        losses = {}
        for mode in ["train", "valid"]:
            loss = _train_vae_encoder_decoder(
                dataloader=dict_dataloaders[mode],
                encoder=encoder,
                decoder=decoder,
                prior=prior,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                mode=mode,
            )
            losses[mode] = loss

        all_scores.append(losses)

        if early_stopping(losses["valid"]):
            logger.info("Training is finished.")
            logger.info("-----")
            break

        if losses["valid"] <= best_loss:
            best_epoch = epoch + 1
            best_loss = losses["valid"]
            logger.info("Best loss is updated.")

            torch.save(encoder.state_dict(), encoder_weight_path)
            torch.save(decoder.state_dict(), decoder_weight_path)

        if epoch % 10 == 0:
            pd.DataFrame(all_scores).to_csv(loss_csv_path, index=False)

        logger.info(f"Elapsed time = {time.time() - _time} sec")
        logger.info("-----")

    pd.DataFrame(all_scores).to_csv(loss_csv_path, index=False)
    end_time = time.time()

    logger.info(f"Best epoch: {best_epoch}, best_loss: {best_loss:.8f}")
    logger.info(f"Train end: {datetime.datetime.utcnow().isoformat()} UTC")
    logger.info(f"Total elapsed time = {(end_time - start_time) / 60.} min")