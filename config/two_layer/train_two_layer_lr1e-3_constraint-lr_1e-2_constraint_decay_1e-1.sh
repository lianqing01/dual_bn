python train_constraint_bn_v2.py --model Two_layer \
    --log_dir two_layer_lr_1e-3_constraint_constraintlr_1e-2_decay_1e-1_exp001 \
    --lr 0.001 \
    --constraint_lr 0.01 \
    --constraint_decay 1 \
    --two_layer
