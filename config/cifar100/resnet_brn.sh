python train.py --model $1 \
    --log_dir $1+_bsz_128 \
    --dataset CIFAR100 \
    --batch-size 128 \
    --lr 0.1 \
