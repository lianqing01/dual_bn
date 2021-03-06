python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/vgg16_constraint_lr1e-1_bsz_128_lr_5e-3_weight_1e-3_noise_ind_2e-9_2e-9 \
    --lr 0.1 \
    --constraint_lr 0.005 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.001 \
    --noise_data_dependent False \
    --noise_mean_std 0.09 \
    --noise_var_std 0.09 \
    --sample_noise True \
    --decrease_affine_lr 0.1 \


