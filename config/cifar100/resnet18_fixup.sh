python train.py --model fixup_resnet18 \
    --log_dir resnet18_fixup_bsz_128 \
    --dataset CIFAR100 \
    --batch-size 128 \
    --grad_clip 1 \
    --lr 0.1 \
