#!/usr/bin/env bash
python train_constraint_bn_v2_1.py \
    --model $1 \
    --log_dir resnet18_constraint_+$1+$2+weight+$3+mean+$4+decay+$5+affine_decay+$6 \
    --norm_layer $2 \
    --lr 0.1 \
    --constraint_lr 0.1 \
    --batch-size $3 \
    --dataset CIFAR100 \
    --constraint_decay $5 \
    --lambda_constraint_weight $4 \
    --dataset $6 \
    ${@:7}
