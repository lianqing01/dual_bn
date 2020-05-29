python train_constraint_bn_v2_1.py --model resnet_constraint56 \
    --log_dir resnet/oracle_resnet_constraint_56_bsz_128 \
    --lr 0.1 \
    --batch-size 128 \
    --constraint_lr 0.005 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.001 \
    --lambda_weight_mean 10 \
    --decrease_affine_lr 0.1 \

