python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2_noaffine \
    --log_dir vgg/vgg16_nobn_bsz_8 \
    --batch-size 8 \
    --lr 0.005 \
    --constraint_lr 0 \
    --lambda_constraint_weight 0
