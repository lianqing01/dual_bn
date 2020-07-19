#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
from tqdm import tqdm
import csv
import os
from comet_ml import Experiment

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import os.path as osp
import torch.optim as optim
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
from torch.utils.tensorboard import SummaryWriter
from utils import progress_bar, AverageMeter
from utils import create_logger
import wandb
from models.constraint_bn_v2 import *
try:
    import torch_xla.core.xla_model as xm
except:
    pass

def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--load_model', type=str, default='')

parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--log_dir', default="oracle_exp001")
parser.add_argument('--grad_clip', default=1)
parser.add_argument('--optim_loss', default="cross_entropy")
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--print_freq', default=10, type=int)



# param for constraint norm
parser.add_argument('--lambda_constraint_weight', default=0, type=float)
parser.add_argument('--constraint_lr', default=0.1, type=float)
parser.add_argument('--constraint_decay', default=1e-3, type=str)
parser.add_argument('--get_optimal_lagrangian',action='store_true', default=False)
parser.add_argument('--decay_constraint', default=-1, type=int)
parser.add_argument('--update_affine_only', default=False, type=str2bool)

# two layer
parser.add_argument('--two_layer', action='store_true', default=False)

# for lr scheduler
parser.add_argument('--lr_ReduceLROnPlateau', default=False, type=str2bool)
parser.add_argument('--schedule', default=[100,150])
parser.add_argument('--decrease_affine_lr', default=1, type=float)
parser.add_argument('--decrease_with_conv_bias', default=False, type=str2bool)
parser.add_argument('--affine_momentum', default=0.9, type=float)
parser.add_argument('--affine_decay', default=1e-4, type=float)

# for adding noise
parser.add_argument('--sample_noise', default=False, type=str2bool)
parser.add_argument('--noise_data_dependent', default=False, type=str2bool)
parser.add_argument('--noise_std', default=0, type=float)
parser.add_argument('--lambda_noise_weight', default=1, type=float)
parser.add_argument('--noise_mean_std', default=0, type=float)
parser.add_argument('--noise_var_std', default=0, type=float)



# dataset
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--add_grad_noise', default=False, type=str2bool)
parser.add_argument('--get_norm_freq', default=1, type=int)



# pretrain
parser.add_argument('--initialize_by_pretrain', action='store_true', default=False)
parser.add_argument('--max_pretrain_epoch', default=20, type=int)
parser.add_argument('--add_noise', default=None, type=str)
parser.add_argument('--lambda_weight_mean', default=1, type=float)


args = parser.parse_args()
args.constraint_decay = float(args.constraint_decay)

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

args.log_dir = args.log_dir
os.makedirs('results/{}'.format(args.log_dir), exist_ok=True)
logger = create_logger('global_logger', "results/{}/log.txt".format(args.log_dir))

wandb.init(project="cifar100", dir="results/{}".format(args.log_dir),
           name=args.log_dir,)
wandb.config.update(args)

#
# Data
logger.info('==> Preparing data..')
if args.augment:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'CIFAR10':
    trainset = datasets.CIFAR10(root='~/data', train=True, download=True,
                            transform=transform_train)
    num_classes=10
elif args.dataset == 'CIFAR100':
    trainset = datasets.CIFAR100(root='~/data', train=True, download=True,
                            transform=transform_train)
    num_classes=100
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True, num_workers=2)

if args.dataset == 'CIFAR10':
    testset = datasets.CIFAR10(root='~/data', train=False, download=True,
                           transform=transform_test)
elif args.dataset == 'CIFAR100':
    testset = datasets.CIFAR100(root='~/data', train=False, download=True,
                            transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

print(num_classes)



logger.info('==> Building model..')
net = models.__dict__[args.model](num_classes=num_classes)
if use_cuda:
    net.cuda()
    logger.info(torch.cuda.device_count())
    cudnn.benchmark = True
    logger.info('Using CUDA..')
else:
    device = xm.xla_device()
    net = net.to(device)

constraint_param = []
for m in net.modules():
    if isinstance(m, Constraint_Lagrangian):
        m.weight_decay = args.constraint_decay
        m.get_optimal_lagrangian = args.get_optimal_lagrangian
        constraint_param.extend(list(map(id, m.parameters())))
affine_param = [p[1] for p in net.named_parameters() if 'mu_' in p[0]]
affine_param.extend([p[1] for p in net.named_parameters() if 'gamma_' in p[0]])
affine_param_id = [id(i) for i in affine_param]

origin_param = filter(lambda p:id(p) not in affine_param_id and id(p) not in constraint_param, net.parameters())
if args.decrease_affine_lr is not None:
    affine_lr = args.decrease_affine_lr * args.lr
else:
    affine_lr = args.lr

optimizer = optim.SGD([
                    {'params': origin_param},
                    {'params':  filter(lambda p:id(p) in constraint_param, net.parameters()),
                        'lr': args.constraint_lr,
                        'weight_decay': args.constraint_decay},
                    {'params': affine_param,
                        'lr': affine_lr,
                        'weight_decay': args.affine_decay,
                        'momentum': args.affine_momentum}
                    ],
                    lr=args.lr, momentum=0.9,
                    weight_decay=args.decay)
'''
constraint_optimizer = (optim.SGD(
                    filter(lambda p:id(p) in constraint_param, net.parameters()),
                    lr=args.constraint_lr, momentum=0.9,
                    weight_decay=args.constraint_decay
                    ))

'''

# Model
if args.resume:
    # Load checkpoint.
    logger.info('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.load_model)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optim'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1

if not os.path.isdir('results/{}'.format(args.log_dir)):
    os.makedirs('results/{}'.format(args.log_dir))
logname = ('results/{}/log_'.format(args.log_dir) + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')

tb_logger = SummaryWriter(log_dir="results/{}".format(args.log_dir))


if args.optim_loss == "cross_entropy":
    criterion = nn.CrossEntropyLoss()
elif args.optim_loss == "mse":
    criterion = nn.MSELoss()
logger.info(args.lr)

if args.update_affine_only == True:
    for m in net.modules():
        if isinstance(m, Constraint_Norm):
            m.update_affine_only = True


if use_cuda:
    device = torch.device("cuda")
print(args.noise_data_dependent)

def train(epoch):
    logger.info('\nEpoch: %d' % epoch)
    net.train()
    train_loss = AverageMeter(100)
    acc = AverageMeter(100)
    batch_time = AverageMeter()
    reg_loss = AverageMeter(100)
    train_loss_avg = 0
    correct = 0
    total = 0
    mean = 0
    var = 0
    lambda_ = 0
    xi_ = 0

    for m in net.modules():
        if isinstance(m, Constraint_Norm):
            m.reset_norm_statistics()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        start = time.time()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        else:
            inputs = inputs.to(device)
            targets = targets.to(device)
        bsz = inputs.size(0)


        outputs = net(inputs)
        if args.optim_loss == 'mse':
            targets = targets.float()
        loss = criterion(outputs, targets)


        # constraint loss
        weight_mean = []
        weight_var = []
        for m in net.modules():
            if isinstance(m, Constraint_Lagrangian):
                weight_mean_, weight_var_ =  m.get_weight_mean_var()
                weight_mean.append(weight_mean_)
                weight_var.append(weight_var_)
        weight_mean = torch.stack(weight_mean).mean()
        weight_var = torch.stack(weight_var).mean()

        constraint_loss = args.lambda_weight_mean * weight_mean + weight_var
        constraint_loss = args.lambda_constraint_weight * constraint_loss

        # optimize constraint loss

        train_loss.update(loss.item())
        train_loss_avg += loss.item()
        loss += constraint_loss

        # optimize
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct_idx = predicted.eq(targets.data).cpu().sum().float()
        correct += correct_idx
        acc.update(100. * correct_idx / float(targets.size(0)))

        optimizer.zero_grad()
        loss.backward()
        if args.add_grad_noise == True:
            for m in net.modules():
                if isinstance(m, Constraint_Norm):
                    m.add_grad_noise_()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        if use_cuda:
            optimizer.step()
        else:
            xm.optimizer_step(optimizer, barrier=True)
        batch_time.update(time.time() - start)
        remain_iter = args.epoch * len(trainloader) - (epoch*len(trainloader) + batch_idx)
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))


        if (batch_idx+1) % args.print_freq == 0:
            logger.info('Train: [{0}][{1}/{2}]\t'
                    'Loss {train_loss.avg:.3f}\t'
                    'acc {acc.avg:.3f}\t'
                    'correct: [{correct}/{total}]\t'
                    'Constraint mean {corat_mean:.4f}\t'
                    'Constraint var {corat_var:.4f}\t'
                    'Constraint lambda {corat_lambda:.4f}\t'
                    'Constraint xi {corat_xi:.4f}\t'
                    'mean {mean:.4f}\t'
                    'var {var:.4f}\t'
                    'remain_time: {remain_time}'.format(
                    epoch, batch_idx, len(trainloader),
                    train_loss = train_loss,
                    corat_mean = -1 * weight_mean.item(),
                    corat_var = -1 * weight_var.item(),
                    corat_lambda = lambda_,
                    corat_xi = xi_,
                    mean = mean,
                    var = var,
                    acc = acc,
                    correct=int(correct),
                    total=total,
                    remain_time=remain_time,
                        ))


        if (batch_idx+1) % args.print_freq == 0:
            mean = []
            var = []
            for m in net.modules():
                if isinstance(m, Constraint_Norm):
                    mean_, var_ = m.get_mean_var()
                    mean.append(mean_.abs())
                    var.append(var_.abs())
            mean = torch.mean(torch.stack(mean))
            var = torch.mean(torch.stack(var))
            curr_idx = epoch * len(trainloader) + batch_idx
            tb_logger.add_scalar("train/train_loss", train_loss.avg, curr_idx)
            tb_logger.add_scalar("train/train_acc", acc.avg, curr_idx)
            tb_logger.add_scalar("train/norm_mean(abs)", mean, curr_idx)
            tb_logger.add_scalar("train/norm_var-1(abs)", var, curr_idx)
            tb_logger.add_scalar("train/constraint_loss_mean", -1 * weight_mean.item(), curr_idx)
            tb_logger.add_scalar("train/constraint_loss_var", -1 * weight_var.item(), curr_idx)

            # get the constraint weight
            lambda_ = []
            xi_ = []
            for m in net.modules():
                if isinstance(m, Constraint_Lagrangian):
                    lambda_.append(m.lambda_.data.abs().mean())
                    xi_.append(m.xi_.data.abs().mean())
            lambda_ = torch.max(torch.stack(lambda_))
            xi_ = torch.max(torch.stack(xi_))
            tb_logger.add_scalar("train/constraint_lambda_", lambda_.item(), curr_idx)
            tb_logger.add_scalar("train/constraint_xi_", xi_.item(), curr_idx)


    tb_logger.add_scalar("train/train_loss_epoch", train_loss_avg / len(trainloader), epoch)
    tb_logger.add_scalar("train/train_acc_epoch", 100.*correct/total, epoch)
    wandb.log({"train/acc_epoch" : 100.*correct/total}, step=epoch)
    wandb.log({"train/loss_epoch" : train_loss_avg/len(trainloader)}, step=epoch)
    wandb.log({"train/norm_mean(abs)": mean.item()}, step=epoch)
    wandb.log({"train/norm_var-1(abs)": var.item()}, step=epoch)
    wandb.log({"train/constraint_loss_mean": -1 * weight_mean.item()}, step=epoch)
    wandb.log({"train/constraint_loss_var": -1 * weight_var.item()},step=epoch)
    logger.info("epoch: {} acc: {}, loss: {}".format(epoch, 100.* correct/total, train_loss_avg / len(trainloader)))

    epoch_mu = []
    epoch_gamma = []
    epoch_mu_error = []
    epoch_gamma_error = []
    real_mus = []
    real_gammas = []
    for m in net.modules():
        if isinstance(m, Constraint_Norm):
            mean_, var_ = m.get_mean_var()
            real_mu, real_gamma = m.get_real_mean_var()
            epoch_mu.append(m.mu_.mean())
            epoch_gamma.append(m.gamma_.mean())
            real_mus.append(real_mu.mean())
            real_gammas.append(real_gamma.mean())
            epoch_mu_error.append( ((torch.abs(m.mu_.view(-1) - real_mu) / (real_mu + 1e-4))).mean())
            epoch_gamma_error.append( (torch.abs(m.gamma_.view(-1).abs() - real_gamma) / (real_gamma + 1e-4 )).mean())
    mu_.append(epoch_mu)
    gamma_.append(epoch_gamma)
    real_mu_.append(real_mus)
    real_gamma_.append(real_gammas)
    mu_error.append(epoch_mu_error)
    gamma_error.append(epoch_gamma_error)
    return (train_loss.avg, reg_loss.avg, 100.*correct/total)


def _initialize(epoch):
    logger.info('\nEpoch: %d' % epoch)
    net.eval()
    train_loss = AverageMeter(100)
    acc = AverageMeter(100)
    batch_time = AverageMeter()
    reg_loss = AverageMeter(100)
    train_loss_avg = 0
    correct = 0
    total = 0
    mean = 0
    var = 0
    lambda_ = 0
    xi_ = 0

    num_norm = 0
    for m in net.modules():
        if isinstance(m, Constraint_Norm):
            m.reset_norm_statistics()
            num_norm+=1
    for layer in range(num_norm):
        for i in range(3):
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                if batch_idx>20 * 128/args.batch_size:
                    break
                start = time.time()
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                else:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                bsz = inputs.size(0)


                outputs = net(inputs)
                if args.optim_loss == 'mse':
                    targets = targets.float()
                loss = criterion(outputs, targets)

                #constraint loss

                weight_mean = []
                weight_var = []
                for m in net.modules():
                    if isinstance(m, Constraint_Lagrangian):
                        weight_mean_, weight_var_ =  m.get_weight_mean_var()
                        weight_mean.append(weight_mean_)
                        weight_var.append(weight_var_)
                weight_mean = torch.stack(weight_mean).mean()
                weight_var = torch.stack(weight_var).mean()


                constraint_loss = weight_mean + weight_var
                constraint_loss = args.lambda_constraint_weight * constraint_loss

                # optimize constraint loss

                train_loss.update(loss.item())
                train_loss_avg += loss.item()
                loss += constraint_loss

                # optimize
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct_idx = predicted.eq(targets.data).cpu().sum().float()
                correct += correct_idx
                acc.update(100. * correct_idx / float(targets.size(0)))

                batch_time.update(time.time() - start)
                remain_iter = args.epoch * len(trainloader) - (epoch*len(trainloader) + batch_idx)
                remain_time = remain_iter * batch_time.avg
                t_m, t_s = divmod(remain_time, 60)
                t_h, t_m = divmod(t_m, 60)
                remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))


                if (batch_idx+1) % args.print_freq == 0:
                    mean = []
                    var = []
                    for m in net.modules():
                        if isinstance(m, Constraint_Norm):
                            mean_, var_ = m.get_mean_var()
                            mean.append(mean_.abs())
                            var.append(var_.abs())
                    mean = torch.mean(torch.stack(mean))
                    var = torch.mean(torch.stack(var))
                    curr_idx = epoch * len(trainloader) + batch_idx

                    # get the constraint weight
                    lambda_ = []
                    xi_ = []
                    for m in net.modules():
                        if isinstance(m, Constraint_Lagrangian):
                            lambda_.append(m.lambda_.data.abs().mean())
                            xi_.append(m.xi_.data.abs().mean())
                    lambda_ = torch.max(torch.stack(lambda_))
                    xi_ = torch.max(torch.stack(xi_))

                if (batch_idx+1) % args.print_freq == 0:
                    logger.info("Initializing")
                    logger.info('Train: [{0}][{1}/{2}]\t'
                            'mean {mean:.4f}\t'
                            'var {var:.4f}\t'
                            'remain_time: {remain_time}'.format(
                            epoch, batch_idx, len(trainloader),
                            mean = mean,
                            var = var,
                            layer=layer,
                            remain_time=remain_time,
                            ))
                    print("layer: {}".format(layer))
            if i == 0:
                track_layer = 0
                for m in net.modules():
                    if isinstance(m, Constraint_Norm):
                        if track_layer == layer:
                            m._initialize_mu()
                            break
                        else:
                            track_layer +=1
            elif i == 1:
                track_layer = 0
                for m in net.modules():
                    if isinstance(m, Constraint_Norm):
                        if track_layer == layer:
                            m._initialize_gamma()
                            break
                        else:
                            track_layer += 1


            for m in net.modules():
                if isinstance(m, Constraint_Norm):
                    m.reset_norm_statistics()


    logger.info("initializing epoch: {} acc: {}, loss: {}".format(epoch, 100.* correct/total, train_loss_avg / len(trainloader)))

    return (train_loss.avg, reg_loss.avg, 100.*correct/total)


def get_norm_stat(epoch):
    logger.info('\nEpoch: %d' % epoch)
    net.train()
    train_loss = AverageMeter(100)
    acc = AverageMeter(100)
    batch_time = AverageMeter()
    reg_loss = AverageMeter(100)
    train_loss_avg = 0
    correct = 0
    total = 0
    mean = 0
    var = 0
    lambda_ = 0
    xi_ = 0


    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        start = time.time()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        else:
            inputs = inputs.to(device)
            targets = targets.to(device)
        bsz = inputs.size(0)


        outputs = net(inputs)
        if args.optim_loss == 'mse':
            targets = targets.float()


        # constraint loss
        weight_mean = []
        weight_var = []
        for m in net.modules():
            if isinstance(m, Constraint_Lagrangian):
                weight_mean_, weight_var_ =  m.get_weight_mean_var()
                weight_mean.append(weight_mean_)
                weight_var.append(weight_var_)
        weight_mean = torch.stack(weight_mean).mean()
        weight_var = torch.stack(weight_var).mean()


        constraint_loss = weight_mean + weight_var
        constraint_loss = args.lambda_constraint_weight * constraint_loss

        # optimize constraint loss
        loss = criterion(outputs, targets)

        loss += constraint_loss
        loss *= args.lr

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)



    optimizer.zero_grad()
    return None




def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    acc2 = AverageMeter(100)
    acc3 = AverageMeter(100)

    for m in net.modules():
        if isinstance(m, Constraint_Norm):
            m.reset_norm_statistics()
    correct = 0
    total = 0
    for m in net.modules():
        if isinstance(m, Constraint_Norm):
            m.reset_norm_statistics()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        else:
            inputs = inputs.to(device)
            targets = targets.to(device)

        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
            bsz = inputs.size(0)
            outputs = net(inputs)
            if args.optim_loss == 'mse':
                targets = targets.float()
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            acc1, acc2_, acc3_ = accuracy(outputs, targets, topk=(1,2,3))

            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            acc2.update(float(acc2_))
            acc3.update(float(acc3_))

        progress_bar(batch_idx, len(testloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total,
                        correct, total))
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
    mean = []
    var = []
    for m in net.modules():
        if isinstance(m, Constraint_Norm):
                mean_, var_ = m.get_mean_var()
                mean.append(mean_.abs())
                var.append(var_.abs())
    mean = torch.mean(torch.stack(mean))
    var = torch.mean(torch.stack(var))


    curr_idx = epoch * len(trainloader)
    tb_logger.add_scalar("test/test_loss", test_loss/batch_idx, curr_idx)
    tb_logger.add_scalar("test/test_acc", 100.*correct/total, curr_idx)
    tb_logger.add_scalar("test/test_loss_epoch", test_loss/batch_idx, epoch)
    tb_logger.add_scalar("test/test_acc_epoch", 100.*correct/total, epoch)

    tb_logger.add_scalar("test/norm_mean(abs)", mean, curr_idx)
    tb_logger.add_scalar("test/norm_var-1(abs)", var, curr_idx)

    tb_logger.add_scalar("test/norm_mean(abs)_epoch", mean, epoch)
    tb_logger.add_scalar("test/norm_var-1(abs)_epoch", var, epoch)



    lambda_ = []
    xi_ = []
    for m in net.modules():
        if isinstance(m, Constraint_Lagrangian):
            lambda_.append(m.lambda_.data.abs().mean())
            xi_.append(m.xi_.data.abs().mean())
    lambda_ = torch.mean(torch.stack(lambda_))
    xi_ = torch.mean(torch.stack(xi_))
    tb_logger.add_scalar("test/constraint_lambda_", lambda_.item(), curr_idx)
    tb_logger.add_scalar("test/constraint_xi_", xi_.item(), curr_idx)
    wandb.log({"test/loss_epoch": test_loss/batch_idx}, step=epoch)
    wandb.log({"test/acc_epoch": 100.*correct/total}, step=epoch)

    wandb.log({"test/norm_mean(abs)_epoch":mean.item()}, step=epoch)
    wandb.log({"test/norm_var-1(abs)_epoch":var.item()}, step=epoch)
    logger.info("acc2: {}".format(acc2.avg))
    logger.info("acc3: {}".format(acc3.avg))




    return (test_loss/batch_idx, 100.*correct/total)

for m in net.modules():
    if isinstance(m, Constraint_Norm):
        m.sample_noise = args.sample_noise
        m.noise_data_dependent = args.noise_data_dependent
        m.noise_std = torch.sqrt(torch.Tensor([args.noise_std])[0].to(device))
        m.sample_mean = torch.zeros(m.num_features).to(device)
        m.add_grad_noise = args.add_grad_noise
        m.lambda_noise_weight = args.lambda_noise_weight
        m.add_noise = args.add_noise
        m.sample_mean_std = torch.sqrt(torch.Tensor([args.noise_mean_std])[0].to(device))
        m.sample_var_std = torch.sqrt(torch.Tensor([args.noise_var_std])[0].to(device))
def checkpoint(acc, epoch):
    # Save checkpoint.
    logger.info('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.makedirs('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7' + args.name + '_'
               + str(args.seed))


def save_checkpoint(acc, epoch):
    logger.info("Saving, epoch: {}".format(epoch))
    state = {
        'state_dict': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'optim': optimizer.state_dict(),
    }
    save_name = osp.join("results/" + args.log_dir, "epoch_{}.pth".format(epoch))
    print(save_name)
    torch.save(state, save_name)


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if args.lr_ReduceLROnPlateau == True:
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, threshold=1e-5,
    )
else:
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones = args.schedule)




if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                            'test loss', 'test acc'])

if args.initialize_by_pretrain == True:
    for epoch in range(args.max_pretrain_epoch):
        mean, var = initialize_by_pretrain(epoch)
        #adjust_learning_rate(optimizer, epoch)
        logger.info("epoch: {} mean: {} var: {}".format(epoch, mean, var))
        if mean + var <=0.2:
            break


if torch.__version__ < '1.4.1':
    lr_scheduler.step(start_epoch)
    lr = optimizer.param_groups[0]['lr']
    logger.info("epoch: {}, lr: {}".format(start_epoch, lr))

if not args.resume:
    with torch.no_grad():
        _initialize(0)
mu_ = []
gamma_ = []
real_mu_ = []
real_gamma_ = []
mu_error = []
gamma_error = []

for epoch in range(start_epoch, args.epoch):
    lr = optimizer.param_groups[0]['lr']
    lr1 = optimizer.param_groups[1]['lr']
    logger.info("begin: epoch: {}, lr: {} lag lr: {}".format(epoch, lr, lr1))

    if epoch == args.decay_constraint:
        args.lambda_constraint_weight = 0

    train_loss, reg_loss, train_acc = train(epoch)
    torch.save([mu_, gamma_, real_mu_, real_gamma_, mu_error, gamma_error], "results/{}/norm_stat.pth".format(args.log_dir))
    if args.lr_ReduceLROnPlateau == True:
        lr_scheduler.step(test_loss)
    else:
        lr_scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    lr1 = optimizer.param_groups[1]['lr']
    logger.info("epoch: {}, lr: {} lag lr: {}".format(epoch, lr, lr1))
    if ((epoch+1) % 10) == 0:
        save_checkpoint(test_acc, epoch)

    test_loss, test_acc = test(epoch)
