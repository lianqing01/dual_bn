#!/usr/bin/env bash

python train_constraint_bn_v2_1_resume.py \
    --model $1 \
    --log_dir noise_$7_$1+$2+weight+$3+mean+$4+decay+$5+affine_decay+$6_lr$8 \
    --norm_layer $2 \
    --lr $8 \
    --constraint_lr 0.01 \
    --batch-size 128 \
    --dataset CIFAR100 \
    --constraint_decay $5 \
    --lambda_constraint_weight $3 \
    --lambda_weight_mean $4 \
    --grad_clip 1 \
    --decrease_affine_lr 1 \
    --affine_decay $6 \
    --sample_noise True \
    --noise_mean_std $7 \
    --noise_var_std $7 \
    ${@:9}
