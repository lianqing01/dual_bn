python train.py --model vgg16_bn \
    --log_dir vgg/oracle_vgg16_bsz_8_bn_Plateau \
    --batch-size 8 \
    --lr 0.005 \
    --lr_ReduceLROnPlateau True
