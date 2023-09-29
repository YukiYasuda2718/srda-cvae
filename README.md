# srda-cvae

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

