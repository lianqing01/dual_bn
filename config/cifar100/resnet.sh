#!/usr/bin/env bash

python train.py --model $1 \
    --log_dir $1+_bn_bsz_+$2_norm_+$4 \
    --dataset CIFAR100 \
    --batch-size $2 \
    --lr $3 \
    --norm_layer $4 \
    ${@:5}
