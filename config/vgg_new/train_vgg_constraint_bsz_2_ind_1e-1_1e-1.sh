python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/vgg16_constraint_bsz_2_noise_ind_1e-1_1e-1 \
    --lr 0.0015625 \
    --constraint_lr 0.00015625 \
    --batch-size 2 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.001 \
    --noise_data_dependent False \
    --noise_mean_std 0.1 \
    --noise_var_std 0.1 \
    --sample_noise True \
    --decrease_affine_lr 0.1 \


