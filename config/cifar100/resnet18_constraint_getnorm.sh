python train_constraint_bn_v2_1.py \
    --model resnet_constraint18 \
    --log_dir vgg/resnet18_constraint_bsz_128_getnorm \
    --lr 0.1 \
    --constraint_lr 0.01 \
    --batch-size 128 \
    --dataset CIFAR100 \
    --constraint_decay 0.01 \
    --lambda_constraint_weight 10 \
    --noise_data_dependent False \
    --lambda_weight_mean 1 \
    --noise_mean_std 0 \
    --noise_var_std 0 \
    --grad_clip 1 \
    --sample_noise True \
    --decrease_affine_lr 1
