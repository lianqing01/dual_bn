python train_pn.py --model resnet50 \
    --log_dir resnet50_lbn_8192_bsz_128 \
    --dataset CIFAR100 \
    --batch-size 128 \
    --pn-batch 8192 \
    --lr 0.1 \
