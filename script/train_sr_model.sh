#!/bin/bash

HOME_DIR=/home/yuki_yasuda/workspace_hub/srda-cvae
ROOT_DIR=/data1/yuki_yasuda/workspace_hub/srda-cvae

export CUDA_VISIBLE_DEVICES="0,1"
WORLD_SIZE=2
SCALE_FACTOR=4

singularity exec \
        --nv \
        --bind $ROOT_DIR:$HOME_DIR \
        --env PYTHONPATH=$HOME_DIR/pytorch \
    $ROOT_DIR/pytorch.sif python3 $HOME_DIR/pytorch/script/train_sr_model_ddp.py \
        --scale_factor $SCALE_FACTOR \
        --world_size $WORLD_SIZE
