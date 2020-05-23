python train_pn.py --model vgg16_pn \
    --log_dir vgg/vgg16_pn_bsz_128_after_x_noise \
    --pn-batch-size 128 \
    --batch-size 128 \
    --sample_noise True \
    --data_dependent after_x \
    --sample_mean_mean 0 \
    --sample_mean_var 1 \
    --sample_std_mean 1e-2 \
    --sample_std_var 1e-1 \
    --batch_renorm False \
    --lr 0.1 \
