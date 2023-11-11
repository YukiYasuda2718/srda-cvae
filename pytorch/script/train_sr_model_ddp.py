import argparse
import copy
import datetime
import json
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
from ml_model.conv2d_sr_net_for_various_scale import ConvSrNet
from src.dataloader import (
    make_dataloaders_vorticity_making_observation_inside_time_series_splitted,
)
from src.loss_maker import MaskedL1Loss
from src.optim_helper import train_prior_model_ddp, validate_prior_model_ddp
from src.utils import set_seeds
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"  # to make calculations deterministic
set_seeds(42, use_deterministic=True)

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--scale_factor", type=int, required=True)
parser.add_argument("--world_size", type=int, required=True)
parser.add_argument("--batch_size", type=int, required=False, default=128)

ROOT_DIR = str((pathlib.Path(os.environ["PYTHONPATH"]) / "..").resolve())
ORG_CONFIG_PATH = f"{ROOT_DIR}/pytorch/config/default_neural_nets.yml"


def get_config(
    scale: int, batch_size: int, epochs: int = 5000, config_path: str = ORG_CONFIG_PATH
) -> dict:
    assert scale % 2 == 0, "Odd scale factors are not supported yet."
    assert 2 <= scale <= 8

    logger.info(f"config path = {config_path}")

    with open(config_path) as file:
        config = yaml.safe_load(file)

    n_blocks = int(np.log2(scale))

    if scale == 6:
        n_blocks = 2

    config["data"]["target_nx"] = 32 * scale
    config["data"]["target_ny"] = 16 * scale
    config["model"]["prior_model"]["num_shuffle_blocks"] = n_blocks
    config["model"]["prior_model"]["scale_factor"] = scale
    config["data"]["scale_factor"] = scale
    config["train"]["first_step"]["epochs"] = epochs
    config["data"]["batch_size"] = batch_size

    if scale == 8:
        config["model"]["prior_model"]["initialization"] = "ICNR"
        config["model"]["prior_model"]["interpolation"] = "bicubic"
    else:
        config["model"]["prior_model"]["interpolation"] = "nearest"

    return config


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

    prior_model = ConvSrNet(**config["model"]["prior_model"])
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

    cleanup()


if __name__ == "__main__":
    try:

        os.environ["MASTER_ADDR"] = "localhost"

        # Port is arbitrary, but set random value to avoid collision
        np.random.seed(datetime.datetime.now().microsecond)
        port = str(np.random.randint(12000, 65535))
        os.environ["MASTER_PORT"] = port

        world_size = parser.parse_args().world_size
        scale_factor = parser.parse_args().scale_factor
        batch_size = parser.parse_args().batch_size

        config = get_config(scale_factor, batch_size)

        os.makedirs(f"{ROOT_DIR}/data/ModelWeights", exist_ok=True)

        if batch_size == 128:
            result_dir_path = f"{ROOT_DIR}/data/ModelWeights/SRx{scale_factor:02}"
        else:
            result_dir_path = f"{ROOT_DIR}/data/ModelWeights/SRx{scale_factor:02}_batch{batch_size:04}"
        os.makedirs(result_dir_path, exist_ok=False)

        logger.addHandler(FileHandler(f"{result_dir_path}/log.txt"))

        logger.info("\n*********************************************************")
        logger.info(f"Start DDP: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

        logger.info(f"Scale factor = {scale_factor}")
        logger.info(f"result dir path = {result_dir_path}")

        if not torch.cuda.is_available():
            logger.error("No GPU.")
            raise Exception("No GPU.")

        logger.info(f"Num available GPUs = {torch.cuda.device_count()}")
        logger.info(f"Names of GPUs = {torch.cuda.get_device_name()}")
        logger.info(f"Device capability = {torch.cuda.get_device_capability()}")
        logger.info(f"World size = {world_size}")

        logger.info("\n*********************************************************")
        logger.info("Config used in this experiment")
        logger.info("*********************************************************")
        logger.info(json.dumps(config, indent=2))
        logger.info("*********************************************************\n")

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