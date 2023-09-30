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
from ml_model.conv2d_sr_v2 import ConvSrNetVer02
from ml_model.conv2d_sr_v3 import ConvSrNetVer03
from ml_model.cvae_snapshot_v2 import CVaeSnapshotVer02
from ml_model.cvae_snapshot_v3 import CVaeSnapshotVer03
from src.dataloader import (
    make_dataloaders_2d_gauss_jet,
    make_dataloaders_vorticity_assimilation,
    make_dataloaders_vorticity_making_observation_inside,
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
set_seeds(42, use_deterministic=True)

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--world_size", type=int, required=True)

ROOT_DIR = str((pathlib.Path(os.environ["PYTHONPATH"]) / "..").resolve())


def check_hr_data_paths_in_dataloader(dataloader):
    for path in dataloader.dataset.hr_paths:
        logger.info(os.path.basename(path))


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

    if config["data"]["data_dir_name"] == "2d_gauss_jet_dt_01p00_v01":
        dataloaders, samplers = make_dataloaders_2d_gauss_jet(
            root_dir=ROOT_DIR, config=config, world_size=world_size, rank=rank
        )
    elif config["data"]["data_dir_name"] in [
        "jet02_obs-intrvl27",
        "jet03_obs-intrvl27",
    ]:
        dataloaders, samplers = make_dataloaders_vorticity_assimilation(
            root_dir=ROOT_DIR, config=config, world_size=world_size, rank=rank
        )
    elif config["data"]["data_dir_name"] == "jet02_obs-intrvl27_not_using_obs":
        dataloaders, samplers = make_dataloaders_vorticity_making_observation_inside(
            ROOT_DIR, "jet02_obs-intrvl27", config, world_size=world_size, rank=rank
        )
    elif (
        config["data"]["data_dir_name"] == "jet12"
        or config["data"]["data_dir_name"] == "jet14"
        or config["data"]["data_dir_name"] == "jet16"
        or config["data"]["data_dir_name"] == "jet18"
    ):
        (
            dataloaders,
            samplers,
        ) = make_dataloaders_vorticity_making_observation_inside_time_series_splitted(
            ROOT_DIR, config, world_size=world_size, rank=rank
        )
    else:
        raise Exception(
            f'datadir name ({config["data"]["data_dir_name"]}) is not supported.'
        )

    if rank == 0:
        # logger.info("\nCheck train_loader")
        # check_hr_data_paths_in_dataloader(dataloaders["train"])
        # logger.info("\nCheck valid_loader")
        # check_hr_data_paths_in_dataloader(dataloaders["valid"])

        logger.info("\n###############################")
        logger.info("Make prior model and optimizer")
        logger.info("###############################\n")

    if config["model"]["prior_model"]["name"] == "ConvSrNetVer02":
        logger.info("Prior model is ConvSrNetVer02")
        prior_model = ConvSrNetVer02(**config["model"]["prior_model"])
    elif config["model"]["prior_model"]["name"] == "ConvSrNetVer03":
        logger.info("Prior model is ConvSrNetVer03")
        prior_model = ConvSrNetVer03(**config["model"]["prior_model"])
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
        torch.save(best_weights, prior_weight_path)
        pd.DataFrame(all_scores).to_csv(prior_learning_history_path, index=False)
        logger.info(f"Best epoch: {best_epoch}, best_loss: {best_loss:.8f}")

    if rank == 0:
        logger.info("\n###############################")
        logger.info("Make cvae model and optimizer")
        logger.info("###############################\n")

    if config["model"]["vae_model"]["name"] == "CVaeSnapshotVer02":
        logger.info("CVaeSnapshotVer02 is created.")
        cvae = CVaeSnapshotVer02(**config["model"]["vae_model"])
    elif config["model"]["vae_model"]["name"] == "CVaeSnapshotVer03":
        logger.info("CVaeSnapshotVer03 is created.")
        cvae = CVaeSnapshotVer03(**config["model"]["vae_model"])
    else:
        raise Exception(f'{config["model"]["vae_model"]["name"]} is not supported.')

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
        torch.save(best_weights, cvae_weight_path)
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

        with open(config_path) as file:
            config = yaml.safe_load(file)

        experiment_name = config_path.split("/")[-2]
        config_name = os.path.basename(config_path).split(".")[0]

        result_dir_path = (
            f"{ROOT_DIR}/data/pytorch/DL_results/{experiment_name}/{config_name}"
        )
        os.makedirs(result_dir_path, exist_ok=False)

        logger.addHandler(FileHandler(f"{result_dir_path}/log.txt"))

        logger.info("\n*********************************************************")
        logger.info(f"Start DDP: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

        logger.info(f"experiment name = {experiment_name}")
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