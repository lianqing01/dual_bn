python train_pn.py --model resnet18 \
    --log_dir resnet18_lbn_2048_bsz_128 \
    --dataset CIFAR100 \
    --batch-size 128 \
    --pn-batch 2048 \
    --lr 0.1 \
