python train.py --model vgg16 \
    --log_dir vgg/oracle_vgg16_bsz_8_nobn_Plateau \
    --batch-size 8 \
    --lr 0.005 \
    --lr_ReduceLROnPlateau True 
