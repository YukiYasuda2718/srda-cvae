#!/bin/bash

HOME_DIR=/home/yuki_yasuda/workspace_hub/srda-cvae
ROOT_DIR=/data1/yuki_yasuda/workspace_hub/srda-cvae

export CUDA_VISIBLE_DEVICES="0,1"
WORLD_SIZE=2
CONFIG_NAME=default_neural_nets.yml

singularity exec \
        --nv \
        --bind $ROOT_DIR:$HOME_DIR \
        --env PYTHONPATH=$HOME_DIR/pytorch \
    $ROOT_DIR/pytorch.sif python3 $HOME_DIR/pytorch/script/train_sr_model_and_cvae_ddp.py \
        --config_path $HOME_DIR/pytorch/config/$CONFIG_NAME \
        --world_size $WORLD_SIZE
