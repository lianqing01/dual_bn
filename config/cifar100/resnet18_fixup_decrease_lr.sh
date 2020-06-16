python train.py --model fixup_resnet18 \
    --log_dir resnet18_fixup_bsz_128_fixup \
    --dataset CIFAR100 \
    --batch-size 128 \
    --lr 0.1 \
    --fixup True \
