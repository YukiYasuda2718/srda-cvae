# srda-cvae  <!-- omit in toc -->

- [Setup](#setup)
  - [Build Docker container](#build-docker-container)
- [Build Singularity container](#build-singularity-container)
- [Make training data](#make-training-data)
- [Train neural networks](#train-neural-networks)
- [Compare results](#compare-results)

## Setup

### Build Docker container

1. Check if `docker-compose` or `docker compose` works.
2. Change preferences in `docker-compose.yml` if needed.
3. Build a container: `$ docker compose build`
4. Run a container: `$ docker compose up -d`


## Build Singularity container

1. Check if `singularity` works.
2. Build a container: `$ singularity build -f pytorch.sif ./singularity/pytorch.def`
3. Change preferences in `./script/start_singularity_container.sh` if needed.
4. Run a container: `$ ./script/start_singularity_container.sh`

## Make training data

- The total data size will be about GB, all of which are stored in `data`.
- Run oceanic jet simulations: `$ ./script/run_jet_simulations.sh`

## Train neural networks

## Compare results

