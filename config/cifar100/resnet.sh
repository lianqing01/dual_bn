python train.py --model $1 \
    --log_dir $1+_bn_bsz_+$2 \
    --dataset CIFAR100 \
    --batch-size $2 \
    --lr $3 \
