import argparse
import copy
import datetime
import os
import pathlib
import sys
import time
import traceback
from logging import INFO, FileHandler, StreamHandler, getLogger

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from ml_model.conv2d_cvae import Conv2dCvae
from ml_model.conv2d_sr_net import ConvSrNet
from src.dataloader import (
    make_dataloaders_vorticity_making_observation_inside_time_series_splitted,
)
from src.loss_maker import MaskedL1Loss, VariationalLowerBoundSnapshot
from src.optim_helper import (
    train_prior_model_ddp,
    train_snapshot_cvae_ddp,
    validate_prior_model_ddp,
    validate_snapshot_cvae_ddp,
)
from src.utils import set_seeds
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"  # to make calculations deterministic
set_seeds(42, use_deterministic=False)

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--world_size", type=int, required=True)

ROOT_DIR = str((pathlib.Path(os.environ["PYTHONPATH"]) / "..").resolve())


def setup(rank: int, world_size: int, backend: str = "nccl"):
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_and_validate(
    rank: int,
    world_size: int,
    config: dict,
    result_dir_path: str,
):

    setup(rank, world_size)
    set_seeds(config["train"]["seed"])

    if rank == 0:
        logger.info("\n###############################")
        logger.info("Make dataloaders and samplers")
        logger.info("################################\n")

    (
        dataloaders,
        samplers,
    ) = make_dataloaders_vorticity_making_observation_inside_time_series_splitted(
        training_data_dir=f"{ROOT_DIR}/data/TrainingData",
        config=config,
        world_size=world_size,
        rank=rank,
    )

    if rank == 0:
        logger.info("\n###############################")
        logger.info("Make prior model and optimizer")
        logger.info("###############################\n")

    if config["model"]["prior_model"]["name"] == "ConvSrNet":
        logger.info("Prior model is ConvSrNet")
        prior_model = ConvSrNet(**config["model"]["prior_model"])
    else:
        raise NotImplementedError(
            f'{config["model"]["prior_model"]["name"]} is not supported.'
        )

    prior_model = DDP(prior_model.to(rank), device_ids=[rank])
    loss_fn = MaskedL1Loss()
    optimizer = torch.optim.Adam(
        prior_model.parameters(), lr=config["train"]["first_step"]["lr"]
    )
    logger.info(f'learning rate = {config["train"]["first_step"]["lr"]}')

    if rank == 0:
        logger.info("\n###############################")
        logger.info("Train prior model")
        logger.info("###############################\n")

    all_scores = []
    best_epoch = 0
    best_loss = np.inf
    best_weights = copy.deepcopy(prior_model.module.state_dict())
    early_stop_count = 0

    prior_weight_path = f"{result_dir_path}/prior_model_weight.pth"
    prior_learning_history_path = f"{result_dir_path}/prior_model_loss.csv"

    for epoch in range(config["train"]["first_step"]["epochs"]):
        _time = time.time()
        dist.barrier()
        loss = train_prior_model_ddp(
            dataloader=dataloaders["train"],
            sampler=samplers["train"],
            prior_model=prior_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
        )
        dist.barrier()
        val_loss = validate_prior_model_ddp(
            dataloader=dataloaders["valid"],
            sampler=samplers["valid"],
            prior_model=prior_model,
            loss_fn=loss_fn,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
        )
        dist.barrier()

        all_scores.append({"loss": loss, "val_loss": val_loss})

        if rank == 0:
            logger.info(
                f"Epoch: {epoch + 1}, loss = {loss:.8f}, val_loss = {val_loss:.8f}"
            )

        if val_loss <= best_loss:
            best_epoch = epoch + 1
            best_loss = val_loss

            logger.info("Best loss is updated.")
            logger.info("Early stopping count is reset")
            early_stop_count = 0

            if rank == 0:
                best_weights = copy.deepcopy(prior_model.module.state_dict())
                torch.save(best_weights, prior_weight_path)
        else:
            early_stop_count += 1
            if (
                early_stop_count
                >= config["train"]["first_step"]["early_stopping_epoch"]
            ):
                logger.info(f"Early stopped. Count = {early_stop_count}")
                break

        if rank == 0:
            if epoch % 10 == 0:
                pd.DataFrame(all_scores).to_csv(
                    prior_learning_history_path, index=False
                )

            logger.info(
                f"Elapsed time = {time.time() - _time} sec, ES count = {early_stop_count}"
            )
            logger.info("-----")

    if rank == 0:
        pd.DataFrame(all_scores).to_csv(prior_learning_history_path, index=False)
        logger.info(f"Best epoch: {best_epoch}, best_loss: {best_loss:.8f}")

    if rank == 0:
        logger.info("\n###############################")
        logger.info("Make cvae model and optimizer")
        logger.info("###############################\n")

    # re-make prior model
    del prior_model
    if config["model"]["prior_model"]["name"] == "ConvSrNet":
        logger.info("Prior model is ConvSrNet")
        prior_model = ConvSrNet(**config["model"]["prior_model"])
    else:
        raise NotImplementedError(
            f'{config["model"]["prior_model"]["name"]} is not supported.'
        )

    prior_model.load_state_dict(torch.load(prior_weight_path))

    # fix all params
    for param in prior_model.parameters():
        param.requires_grad = False
    _ = prior_model.eval()

    if config["model"]["vae_model"]["name"] == "Conv2dCvae":
        logger.info("Conv2dCvae is created.")
        cvae = Conv2dCvae(prior_model=prior_model, **config["model"]["vae_model"])
    else:
        raise Exception(f'{config["model"]["vae_model"]["name"]} is not supported.')

    prior_model = prior_model.to(rank)
    cvae = DDP(cvae.to(rank), device_ids=[rank])
    vlb_fn = VariationalLowerBoundSnapshot(**config["train"]["second_step"]["loss"])
    optimizer = torch.optim.Adam(
        cvae.parameters(), lr=config["train"]["second_step"]["lr"]
    )
    logger.info(f'learning rate = {config["train"]["second_step"]["lr"]}')

    if rank == 0:
        logger.info("\n###############################")
        logger.info("Train cvae model")
        logger.info("###############################\n")

    all_scores = []
    best_epoch = 0
    best_loss = np.inf
    best_weights = copy.deepcopy(cvae.module.state_dict())
    early_stop_count = 0

    cvae_weight_path = f"{result_dir_path}/cvae_weight.pth"
    cvae_learning_history_path = f"{result_dir_path}/cvae_loss.csv"

    for epoch in range(config["train"]["second_step"]["epochs"]):
        _time = time.time()
        dist.barrier()
        loss = train_snapshot_cvae_ddp(
            dataloader=dataloaders["train"],
            sampler=samplers["train"],
            cvae=cvae,
            prior_model=prior_model,
            vlb_fn=vlb_fn,
            optimizer=optimizer,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
        )
        dist.barrier()
        val_loss = validate_snapshot_cvae_ddp(
            dataloader=dataloaders["valid"],
            sampler=samplers["valid"],
            cvae=cvae,
            prior_model=prior_model,
            vlb_fn=vlb_fn,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
        )
        dist.barrier()

        all_scores.append({"loss": loss, "val_loss": val_loss})

        if rank == 0:
            logger.info(
                f"Epoch: {epoch + 1}, loss = {loss:.8f}, val_loss = {val_loss:.8f}"
            )

        if val_loss <= best_loss:
            best_epoch = epoch + 1
            best_loss = val_loss

            logger.info("Best loss is updated.")
            logger.info("Early stopping count is reset")
            early_stop_count = 0

            if rank == 0:
                best_weights = copy.deepcopy(cvae.module.state_dict())
                torch.save(best_weights, cvae_weight_path)
        else:
            early_stop_count += 1
            if (
                early_stop_count
                >= config["train"]["second_step"]["early_stopping_epoch"]
            ):
                logger.info(f"Early stopped. Count = {early_stop_count}")
                break

        if rank == 0:
            if epoch % 10 == 0:
                pd.DataFrame(all_scores).to_csv(cvae_learning_history_path, index=False)

            logger.info(
                f"Elapsed time = {time.time() - _time} sec, ES count = {early_stop_count}"
            )
            logger.info("-----")

    if rank == 0:
        pd.DataFrame(all_scores).to_csv(cvae_learning_history_path, index=False)
        logger.info(f"Best epoch: {best_epoch}, best_loss: {best_loss:.8f}")

    cleanup()


if __name__ == "__main__":
    try:

        os.environ["MASTER_ADDR"] = "localhost"

        # Port is arbitrary, but set random value to avoid collision
        np.random.seed(datetime.datetime.now().microsecond)
        port = str(np.random.randint(12000, 65535))
        os.environ["MASTER_PORT"] = port

        world_size = parser.parse_args().world_size
        config_path = parser.parse_args().config_path

        os.makedirs(f"{ROOT_DIR}/data/ModelWeights", exist_ok=True)

        with open(config_path) as file:
            config = yaml.safe_load(file)

        config_name = os.path.basename(config_path).split(".")[0]

        result_dir_path = f"{ROOT_DIR}/data/ModelWeights/{config_name}"
        os.makedirs(result_dir_path, exist_ok=False)

        logger.addHandler(FileHandler(f"{result_dir_path}/log.txt"))

        logger.info("\n*********************************************************")
        logger.info(f"Start DDP: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

        logger.info(f"config name = {config_name}")
        logger.info(f"config path = {config_path}")

        if not torch.cuda.is_available():
            logger.error("No GPU.")
            raise Exception("No GPU.")

        logger.info(f"Num available GPUs = {torch.cuda.device_count()}")
        logger.info(f"Names of GPUs = {torch.cuda.get_device_name()}")
        logger.info(f"Device capability = {torch.cuda.get_device_capability()}")
        logger.info(f"World size = {world_size}")

        start_time = time.time()

        mp.spawn(
            train_and_validate,
            args=(world_size, config, result_dir_path),
            nprocs=world_size,
            join=True,
        )

        end_time = time.time()

        logger.info(f"Total elapsed time = {(end_time - start_time) / 60.} min")

        logger.info("\n*********************************************************")
        logger.info(f"End DDP: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

    except Exception as e:
        logger.info("\n*********************************************************")
        logger.info("Error")
        logger.info("*********************************************************\n")
        logger.error(e)
        logger.error(traceback.format_exc())