python train_pn.py --model resnet20 \
    --log_dir cifar100/resnet20_lbn_bsz_8192 \
    --pn-batch-size 8192 \
    --batch-size 128 \
    --dataset CIFAR100
    --lr 0.1 \
