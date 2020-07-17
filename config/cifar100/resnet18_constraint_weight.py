import subprocess
import argparse
import time
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default="resnet_constraint18", type=str, help='learning rate')
parser.add_argument('--gpus', default=[0,1,2,3], type=list)
parser.add_argument('--constraint_weight', default=[10, 1, 0.1], type=list)


args = parser.parse_args()

for i in range(len(args.constraint_weight)):
    script = "CUDA_VISIBLE_DEVICES={} python train_constraint_bn_v2_1.py \
    --model {} \
    --log_dir vgg/resnet50_constraint_bsz_128_weight_{} \
    --lr 0.1 \
    --constraint_lr 0.01 \
    --batch-size 128 \
    --dataset CIFAR100 \
    --constraint_decay 1 \
    --lambda_constraint_weight {} \
    --noise_data_dependent False \
    --lambda_weight_mean 1 \
    --grad_clip 1 \
    --sample_noise False \
    --decrease_affine_lr 1".format(args.gpus[i], args.model, args.constraint_weight[i], args.constraint_weight[i])
    print(script)
    gpu_script = "export CUDA_VISIBLE_DEVICES={}".format(args.gpus[i])
    subprocess.Popen(gpu_script, shell=True)
    print(gpu_script)
    time.sleep(2)
    subprocess.Popen(script, stdin=None, stdout=None, stderr=None, shell=True)
