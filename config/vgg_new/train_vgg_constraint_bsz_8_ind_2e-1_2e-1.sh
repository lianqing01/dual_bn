python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/vgg16_constraint_bsz_8_noise_ind_2e-1_2e-1 \
    --lr 0.00625 \
    --constraint_lr 0.000625 \
    --batch-size 8 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.001 \
    --noise_data_dependent False \
    --noise_mean_std 0.01 \
    --noise_var_std 0.01 \
    --sample_noise True \
    --decrease_affine_lr 0.1 \


