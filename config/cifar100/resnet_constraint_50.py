import subprocess
import argparse
import time
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default="resnet_constraint18", type=str, help='learning rate')
parser.add_argument('--gpus', default=[0,1,2,3], type=list)
parser.add_argument('--noise', default=[1e-1, 1e-2, 5e-2, 1e-3], type=list)


args = parser.parse_args()

for i in range(len(args.noise)):
    script = "CUDA_VISIBLE_DEVICES={} python train_constraint_bn_v2_1.py \
    --model {} \
    --log_dir vgg/resnet18_constraint_bsz_128_noise_ind_{} \
    --lr 0.1 \
    --constraint_lr 0.001 \
    --batch-size 128 \
    --dataset CIFAR100 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0 \
    --noise_data_dependent False \
    --lambda_weight_mean 1 \
    --noise_mean_std {} \
    --noise_var_std {} \
    --grad_clip 1 \
    --sample_noise True \
    --decrease_affine_lr 1".format(args.gpus[i], args.model, args.noise[i], args.noise[i], args.noise[i])
    print(script)
    gpu_script = "export CUDA_VISIBLE_DEVICES={}".format(args.gpus[i])
    subprocess.Popen(gpu_script, shell=True)
    print(gpu_script)
    time.sleep(2)
    subprocess.Popen(script, stdin=None, stdout=None, stderr=None, shell=True)
