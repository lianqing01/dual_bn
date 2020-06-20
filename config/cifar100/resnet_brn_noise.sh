python train.py --model $1 \
    --log_dir resnet18_brn_noise_bsz_128+$2 \
    --dataset CIFAR100 \
    --batch-size 128 \
    --sample_noise True \
    --noise_std_mean $2 \
    --noise_std_var $2 \
    --lr 0.1 \
