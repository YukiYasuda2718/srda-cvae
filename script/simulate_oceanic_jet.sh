#!/bin/bash

HOME_DIR=/home/yuki_yasuda/workspace_hub/srda-cvae
ROOT_DIR=/data1/yuki_yasuda/workspace_hub/srda-cvae

export CUDA_VISIBLE_DEVICES="0"

I_SEED_START=0
I_SEED_END=249

singularity exec \
      --nv \
      --bind $ROOT_DIR:$HOME_DIR \
      --env PYTHONPATH=$HOME_DIR/pytorch \
    $ROOT_DIR/pytorch.sif python3 $HOME_DIR/pytorch/script/simulate_oceanic_jet.py \
        --i_seed_start $I_SEED_START --i_seed_end $I_SEED_END
