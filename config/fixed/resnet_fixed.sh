#!/usr/bin/env bash

python train_fixed_weight.py --model $1 \
    --log_dir $1+$2 \
    --trained_weight $2 \
    --lr $3 \
    --norm_layer $4 \
    ${@:5}

