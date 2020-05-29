python train_constraint_bn_v2_1.py --model resnet_constraint56 \
    --log_dir resnet/oracle_resnet_constraint_56_bsz_128_4 \
    --lr 0.1 \
    --batch-size 128 \
    --constraint_lr 0.01 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.0005 \
    --lambda_weight_mean 1 \
    --decrease_affine_lr 0.1 \

