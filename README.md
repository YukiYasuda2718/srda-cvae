#  Super-Resolution Data Assimilation (SRDA) Using Conditional Variational Autoencoders <!-- omit in toc -->

This repository contains the code used in "*Theory of Super-Resolution Data Assimilation with Conditional Variational Autoencoders*" ([arXiv](https://arxiv.org/abs/2308.03351)).

- [Setup](#setup)
  - [Build Singularity container](#build-singularity-container)
  - [Build Docker container](#build-docker-container)
- [Run experiments](#run-experiments)
  - [Make training data](#make-training-data)
  - [Train neural networks](#train-neural-networks)
  - [Compare results](#compare-results)
- [Citation](#citation)

## Setup

- The experiments have been conducted in the Singularity container.
- The same experimental environment can be made using Docker.

### Build Singularity container

- The `sif` file size is about 7 GB.

1. Check if `singularity` works.
2. Build a container: `$ singularity build -f pytorch.sif ./singularity/pytorch.def`
3. Change preferences in `./script/start_singularity_container.sh` if needed.
4. Run a container: `$ ./script/start_singularity_container.sh`

### Build Docker container

1. Check if `docker compose` or `docker-compose` works.
2. Change preferences in `docker-compose.yml` if needed.
3. Build a container: `$ docker compose build` or `$ docker-compose build`
4. Run a container: `$ docker compose up -d` or `$ docker-compose up -d`

## Run experiments

- The experiments have been conducted in the Singularity container.
- The container is started using [`./script/start_singularity_container.sh`](./script/start_singularity_container.sh).
  - The preferences, such as port numbers (`PORT`), need to be modified first.
- The total data size will be about 60 GB, all of which are stored in `data`.

### Make training data

1. Set preferences, such as `HOME_DIR`, in [`./script/simulate_oceanic_jet.sh`](./script/simulate_oceanic_jet.sh).
2. Run oceanic jet simulations: `$ ./script/simulate_oceanic_jet.sh`
3. Start the Singularity container: `$ ./script/start_singularity_container.sh`
4. Connect to the JupyterLab inside the container.
5. Split simulation results by running [this notebook](./pytorch/notebook/split_jet_simulation_results.ipynb).

### Train neural networks

1. Set preferences, such as `HOME_DIR`, in [`./script/train_sr_model_and_cvae.sh`](./script/train_sr_model_and_cvae.sh).
2. Run the training script: `$ ./script/train_sr_model_and_cvae.sh`

### Compare results

1. Start the Singularity container: `$ ./script/start_singularity_container.sh`
2. Connect to the JupyterLab inside the container.
3. Run SRDA using [this notebook](./pytorch/notebook/run_srda.ipynb).
4. Run EnKF using [this notebook](./pytorch/notebook/run_enkf.ipynb).
5. Plot the results using [this notebook](./pytorch/notebook/plot_results.ipynb).

## Citation

```bibtex
@misc{
  title={Theory of Super-Resolution Data Assimilation with Conditional Variational Autoencoders}, 
  author={Yuki Yasuda and Ryo Onishi},
  year={2024},
  eprint={2308.03351},
  archivePrefix={arXiv},
  primaryClass={physics.ao-ph},
  url={https://arxiv.org/abs/2308.03351}, 
}
```
