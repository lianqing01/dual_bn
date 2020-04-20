python train_constraint_bn_v2.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/oracle_vgg16_bsz_8_constraint_constraint_lr_5e-4_weight_1_512_decay_weight \
    --lr 0.005 \
    --batch-size 8 \
    --constraint_lr 0.0005 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.001953 \
    --decay_weight 30

