#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function
from comet_ml import Experiment

import argparse
from models.batchrenorm import BatchRenorm2d
import os.path as osp
import time
import csv
import os
try:
    import torch_xla.core.xla_model as xm
except:
    pass

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb
import models
from torch.utils.tensorboard import SummaryWriter
from utils import progress_bar, AverageMeter
from utils import create_logger
from models.batchnorm import BatchNorm2d
from models.group_norm import GroupNorm

def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            return v

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
parser.add_argument('--pn-batch-size', default=128, type=int, help='batch size')

parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--log_dir', default="oracle_exp001")
parser.add_argument('--grad_clip', default=1)
# for lr scheduler
parser.add_argument('--lr_ReduceLROnPlateau', default=False, type=str2bool)
parser.add_argument('--schedule', default=[100,150])
parser.add_argument('--fixup', default=False)
parser.add_argument('--decrease_affine', default=False)
parser.add_argument('--sample_noise', default=False, type=str2bool)
parser.add_argument('--data_dependent', default=False, type=str2bool)
parser.add_argument('--noise_bsz', default=128, type=int)
parser.add_argument('--noise_std', default=0, type=float)
parser.add_argument('--r_max', default=0.5, type=float)
parser.add_argument('--batch_renorm', default=False, type=str2bool)
parser.add_argument('--project_name', default="dual_bn_v2", type=str)


# dataset
parser.add_argument('--dataset', default='CIFAR10', type=str)

parser.add_argument('--print_freq', default=10, type=int)

# for noise
parser.add_argument('--noise_mean_mean', default=0, type=float)
parser.add_argument('--noise_mean_var', default=0, type=float)
parser.add_argument('--noise_mean_std', default=0, type=float)
parser.add_argument('--noise_var_std', default=0, type=float)
parser.add_argument('--after_x', default="version_1", type=str)



args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

torch.manual_seed(args.seed)

args.log_dir = args.log_dir + '_' + time.asctime(time.localtime(time.time())).replace(" ", "-")
os.makedirs('results/{}'.format(args.log_dir), exist_ok=True)
logger = create_logger('global_logger', "results/{}/log.txt".format(args.log_dir))



wandb.init(project=args.project_name, dir="results/{}".format(args.log_dir),
           name=args.log_dir,)
wandb.config.update(args)

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
                                          drop_last=True,
                                          shuffle=True, num_workers=8)
if args.pn_batch_size - args.batch_size > 0:
    pn_trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.pn_batch_size - args.batch_size,
                                          drop_last=True,
                                          shuffle=True, num_workers=8)


if args.dataset == 'CIFAR10':
    testset = datasets.CIFAR10(root='~/data', train=False, download=False,
                           transform=transform_test)
elif args.dataset == 'CIFAR100':
    testset = datasets.CIFAR100(root='~/data', train=False, download=True,
                            transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=4)


net = models.__dict__[args.model](num_classes=num_classes)
# Model


if use_cuda:
    net.cuda()
    #net = torch.nn.DataParallel(net)
    logger.info(torch.cuda.device_count())
    cudnn.benchmark = True
    logger.info('Using CUDA..')
else:
    device = xm.xla_device()
    net = net.to(device)
    logger.info("xla")

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)

if args.resume:
    # Load checkpoint.
    logger.info('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.load_model)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optim'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1

else:
    logger.info('==> Building model..')
logname = ('results/{}/log_'.format(args.log_dir) + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')

tb_logger = SummaryWriter(log_dir="results/{}".format(args.log_dir))

criterion = nn.CrossEntropyLoss()
logger.info(args.lr)
#wandb.watch(net)

if args.fixup:
    parameters_bias = [p[1] for p in net.named_parameters() if 'bias' in p[0]]
    parameters_scale = [p[1] for p in net.named_parameters() if 'scale' in p[0]]
    parameters_others = [p[1] for p in net.named_parameters() if not ('bias' in p[0] or 'scale' in p[0])]
    optimizer = optim.SGD(
            [{'params': parameters_bias, 'lr': args.lr/10.},
            {'params': parameters_scale, 'lr': args.lr/10.},
            {'params': parameters_others}],
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.decay)

if args.decrease_affine:
    affine_param = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            affine_param.extend(list(map(id, m.bias)))
    origin_param = filter(lambda p:id(p) not in affine_param, net.parameters())

    optimizer = optim.SGD([
                       {'params': origin_param},
                       {'params': filter(lambda p:id(p) in affine_param, net.parameters()),
                            'lr': args.lr/10.}
                       ],
                      lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)






def train(epoch):
    logger.info('\nEpoch: %d' % epoch)
    net.train()
    train_loss = AverageMeter(100)
    reg_loss = AverageMeter(100)
    train_loss_avg = 0
    correct = 0
    total = 0
    acc = AverageMeter(100)
    batch_time = AverageMeter()
    if args.pn_batch_size - args.batch_size > 0:
        pn_iter = iter(pn_trainloader)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        start = time.time()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        else:
            inputs = inputs.to(device)
            targets = targets.to(device)
        if args.pn_batch_size - args.batch_size > 0:
            try:
                inputs_pn, targets_pn = pn_iter.__next__()
            except:
                pn_iter = iter(pn_trainloader)
                inputs_pn, targets_pn = pn_iter.__next__()
            if use_cuda:
                inputs_pn = inputs_pn.cuda()
                targets_pn = targets_pn.cuda()
            else:
                inputs_pn = inputs_pn.to(device)
                targets_pn = targets_pn.to(device)
            inputs = torch.cat([inputs, inputs_pn],dim=0)
            targets = torch.cat([targets, targets_pn], dim=0)



        outputs = net(inputs)
        loss = criterion(outputs[:args.batch_size], targets[:args.batch_size])
        train_loss.update(loss.data.item())
        _, predicted = torch.max(outputs[:args.batch_size].data, 1)
        total += args.batch_size
        correct_idx = predicted.eq(targets[:args.batch_size].data).cpu().sum().float()
        correct += correct_idx
        acc.update(100. * correct_idx / float(args.batch_size))
        train_loss_avg += loss.item()

        optimizer.zero_grad()
        loss.backward()
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
                    '[{correct}/{total}]\t'
                    'remain_time: {remain_time}'.format(
                    epoch, batch_idx, len(trainloader),
                    train_loss = train_loss,
                    acc = acc,
                    correct=int(correct),
                    total=total,
                    remain_time=remain_time,
                        ))

        if (batch_idx+1) % args.print_freq == 0:
            curr_idx = epoch * len(trainloader) + batch_idx
            tb_logger.add_scalar("train/train_loss", train_loss.avg, curr_idx)
            tb_logger.add_scalar("train/train_acc", acc.avg, curr_idx)
            #experiment.log_metric("loss_step", train_loss.avg, curr_idx)
            #experiment.log_metric("acc_step", acc.avg, curr_idx)
            #wandb.log({"train_loss": train_loss.avg}, step=curr_idx)
            #wandb.log({"train_acc":acc.avg}, step=curr_idx)



    tb_logger.add_scalar("train/train_loss_epoch", train_loss_avg / len(trainloader), epoch)
    tb_logger.add_scalar("train/train_acc_epoch", 100.*correct/total, epoch)
    wandb.log({"train/acc_epoch" : 100.*correct/total}, step=epoch)
    wandb.log({"train/loss_epoch" : train_loss_avg/len(trainloader)}, step=epoch)

    logger.info("epoch: {} acc: {}, loss: {}".format(epoch, 100.* correct/total, train_loss_avg / len(trainloader)))
    return (train_loss.avg, reg_loss.avg, 100.*correct/total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = AverageMeter(100)
    acc = AverageMeter(100)
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        else:
            inputs = inputs.to(device)
            targets = targets.to(device)


        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss.update(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct_idx = predicted.eq(targets.data).sum().item()
            correct += correct_idx

            acc.update(100. * correct_idx / float(targets.size(0)))
        progress_bar(batch_idx, len(testloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss.avg, acc.avg,
                        correct, total))
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
    tb_logger.add_scalar("test/test_loss", test_loss.avg, epoch * len(trainloader))
    tb_logger.add_scalar("test/test_acc", 100.*correct/total, epoch*len(trainloader))
    tb_logger.add_scalar("test/test_loss_epoch", test_loss.avg, epoch)
    tb_logger.add_scalar("test/test_acc_epoch", 100.*correct/total, epoch)
    wandb.log({"test/loss_epoch": test_loss.avg}, step=epoch)
    wandb.log({"test/acc_epoch": 100.*correct/total}, step=epoch)

    return (test_loss.avg, 100.*correct/total)



def save_checkpoint(acc, epoch):
    logger.info("Saving, epoch: {}".format(epoch))
    state = {
        'state_dict': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'optim': optimizer.state_dict(),
    }
    save_name = osp.join("results/" + args.log_dir, "epoch_{}.pth".format(epoch))
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

if torch.__version__ < '1.4.1':
    lr_scheduler.step(start_epoch)
    lr = optimizer.param_groups[0]['lr']
    logger.info("epoch: {}, lr: {}".format(start_epoch, lr))

if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                            'test loss', 'test acc'])

if use_cuda:
    device = torch.device("cuda")
print(args.data_dependent)

for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    if args.lr_ReduceLROnPlateau == True:
        lr_scheduler.step(test_loss)
    else:
        lr_scheduler.step()

    lr = optimizer.param_groups[0]['lr']
    logger.info("epoch: {}, lr: {}".format(epoch, lr))
    if ((epoch+1) % 10) == 0:
        save_checkpoint(test_acc, epoch)

    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                            test_acc])
