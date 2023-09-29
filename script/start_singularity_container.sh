#!/bin/bash

PORT=10168
HOME_DIR=/home/yuki_yasuda/workspace_hub/srda-cvae
ROOT_DIR=/data1/yuki_yasuda/workspace_hub/srda-cvae

export CUDA_VISIBLE_DEVICES="0,1,2,3"

singularity exec \
        --nv \
        --bind $ROOT_DIR:$HOME_DIR \
        --env PYTHONPATH=$HOME_DIR/pytorch \
    $ROOT_DIR/pytorch.sif jupyter lab \
        --no-browser --ip=0.0.0.0 --allow-root --LabApp.token='' --port=$PORT
