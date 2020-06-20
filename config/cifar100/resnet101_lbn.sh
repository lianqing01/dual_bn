python train_pn.py --model resnet101 \
    --log_dir resnet101_lbn_128_bsz_128 \
    --dataset CIFAR100 \
    --batch-size 128 \
    --pn-batch $1 \
    --lr 0.1 \
