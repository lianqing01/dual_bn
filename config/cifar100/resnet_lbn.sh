#!/usr/bin/env bash

python train_pn.py --model $1 \
    --log_dir $1+_bn_bsz_+$2_pn_$3+_norm_+$5_noise_$6 \
    --dataset CIFAR100 \
    --batch-size $2 \
    --pn-batch $3 \
    --lr $4 \
    --norm_layer $5 \
    --sample_noise True \
    --noise_std_mean $6 \
    --noise_std_var $6 \
    ${@:7}
