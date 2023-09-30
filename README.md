# srda-cvae  <!-- omit in toc -->

- [Setup](#setup)
  - [Build Singularity container](#build-singularity-container)
  - [Build Docker container](#build-docker-container)
- [Run experiments](#run-experiments)
  - [Make training data](#make-training-data)
  - [Train neural networks](#train-neural-networks)
  - [Compare results](#compare-results)

## Setup

- The experiments have been conducted in the Singularity container.
- The same experimental environment can be made using Docker.

### Build Singularity container

1. Check if `singularity` works.
2. Build a container: `$ singularity build -f pytorch.sif ./singularity/pytorch.def`
3. Change preferences in `./script/start_singularity_container.sh` if needed.
4. Run a container: `$ ./script/start_singularity_container.sh`

### Build Docker container

1. Check if `docker-compose` or `docker compose` works.
2. Change preferences in `docker-compose.yml` if needed.
3. Build a container: `$ docker compose build`
4. Run a container: `$ docker compose up -d`

## Run experiments

- The experiments have been conducted in the Singularity container.

### Make training data

- The total data size will be about 50 GB, all of which are stored in `data`.

1. Set preferences, such as `HOME_DIR`, in [`./script/run_jet_simulations.sh`](./script/run_jet_simulations.sh).
2. Run oceanic jet simulations: `$ ./script/run_jet_simulations.sh`
3. Split simulation results by running [this notebook](./pytorch/notebook/split_jet_simulation_results.ipynb).

### Train neural networks

1. Set preferences, such as `HOME_DIR`, in [`./script/train_neural_nets.sh`](./script/train_neural_nets.sh).
2. Run oceanic jet simulations: `$ ./script/train_neural_nets.sh`

### Compare results

1. Run SRDA using [this notebook](./pytorch/notebook/run_srda.ipynb).
2. Run EnKF using [this notebook](./pytorch/notebook/run_enkf.ipynb).
3. Compare the results using this notebook.
