python train.py --model fixup_resnet101 \
    --log_dir resnet101_fixup_bsz_128 \
    --dataset CIFAR100 \
    --batch-size 128 \
    --fixup True \
    --grad_clip 1 \
    --lr 0.1 \
