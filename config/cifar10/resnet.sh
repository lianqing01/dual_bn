python train.py --model $1 \
    --log_dir cifar10+$1+_bn_bsz_+$2_norm_+$4 \
    --dataset CIFAR10 \
    --batch-size $2 \
    --lr $3 \
    --norm_layer $4 \
