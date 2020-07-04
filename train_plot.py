#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function
from comet_ml import Experiment

import argparse
import os.path as osp
import time
from models import BatchNorm_augmented2d
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

def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
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
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--log_dir', default="oracle_exp001")
parser.add_argument('--test', default=False, type=bool)
parser.add_argument('--grad_clip', default=10)
# for lr scheduler
parser.add_argument('--lr_ReduceLROnPlateau', default=False, type=bool)
parser.add_argument('--schedule', default=[100,150])
parser.add_argument('--fixup', default=False)
parser.add_argument('--decrease_affine', default=False)
parser.add_argument('--fixup_scale_decay', default=1e-4, type=float)
parser.add_argument('--bn_param_lr', default=0.4, type=float)
parser.add_argument('--lag_param_lr', default=0.01, type=float)



# dataset
parser.add_argument('--dataset', default='CIFAR10', type=str)

parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--sample_noise', default=False, type=str2bool)
parser.add_argument('--noise_std_mean', default=0, type=float)
parser.add_argument('--noise_std_var', default=0, type=float)
parser.add_argument('--norm_layer', default=None, type=str)




args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

torch.manual_seed(args.seed)

args.log_dir = args.log_dir + '_' + time.asctime(time.localtime(time.time())).replace(" ", "-")
os.makedirs('results/{}'.format(args.log_dir), exist_ok=True)
logger = create_logger('global_logger', "results/{}/log.txt".format(args.log_dir))



wandb.init(project="cifar100", dir="results/{}".format(args.log_dir),
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
                                          shuffle=True, num_workers=4)

if args.dataset == 'CIFAR10':
    testset = datasets.CIFAR10(root='~/data', train=False, download=False,
                           transform=transform_test)
elif args.dataset == 'CIFAR100':
    testset = datasets.CIFAR100(root='~/data', train=False, download=True,
                            transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=4)


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
if args.norm_layer is not None and args.norm_layer != 'False':
    if args.norm_layer == 'mv_v0':
        norm_layer = models.__dict__['BatchNorm_mv2d']
    elif args.norm_layer == 'mv_v1':
        norm_layer = models.__dict__['BatchNorm_mvv12d']
    elif args.norm_layer == 'mv_v2':
        norm_layer = models.__dict__['BatchNorm_mvv22d']
    elif args.norm_layer == 'mv_aug':
        norm_layer = models.__dict__['BatchNorm_augmented2d']
    else:
        norm_layer = nn.BatchNorm2d
    net = models.__dict__[args.model](num_classes=num_classes, norm_layer=norm_layer)
else:
    net = models.__dict__[args.model](num_classes=num_classes)

# Model
if args.resume:
    # Load checkpoint.
    logger.info('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.load_model)
    net.load_state_dict(checkpoint['state_dict'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
else:
    logger.info('==> Building model..')


logname = ('results/{}/log_'.format(args.log_dir) + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')

tb_logger = SummaryWriter(log_dir="results/{}".format(args.log_dir))

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

criterion = nn.CrossEntropyLoss()
logger.info(args.lr)
#wandb.watch(net)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)

if args.fixup:
    parameters_bias = [p[1] for p in net.named_parameters() if 'bias' in p[0]]
    parameters_scale = [p[1] for p in net.named_parameters() if 'scale' in p[0]]
    parameters_others = [p[1] for p in net.named_parameters() if not ('bias' in p[0] or 'scale' in p[0])]
    optimizer = optim.SGD(
            [{'params': parameters_bias, 'lr': args.lr/10., 'weight_decay': args.fixup_scale_decay},
            {'params': parameters_scale, 'lr': args.lr/10., 'weight_decay': args.fixup_scale_decay},
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

if args.norm_layer == 'mv_aug':
    parameter_mu = [p[1] for p in net.named_parameters() if 'mu_' in p[0]]
    parameter_gamma = [p[1] for p in net.named_parameters() if 'gamma_' in p[0]]
    parameter_lag = [p[1] for p in net.named_parameters() if ('alpha' in p[0] or 'beta' in p[0])]

    parameter_others = [p[1] for p in net.named_parameters() if not ('mu_' in p[0] or 'gamma_' in p[0] or \
                                                                     'alpha' in p[0] or 'beta' in p[0])]

    optimizer = optim.SGD(
        [{'params': parameter_mu, 'lr': args.bn_param_lr, 'weight_decay': 0, 'momentum': 0},
         {'params': parameter_gamma, 'lr': args.bn_param_lr, 'weight_decay': 0, 'momentum': 0},
         {'params': parameter_lag, 'lr': args.lag_param_lr, 'weight_decay': 0, 'momentum': 0.9},
         {'params': parameter_others}],
        lr=args.lr,
        momentum=0.9,
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
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        start = time.time()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        else:
            inputs = inputs.to(device)
            targets = targets.to(device)


        outputs = net(inputs)
        loss = criterion(outputs, targets)
        train_loss.update(loss.data.item())
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct_idx = predicted.eq(targets.data).cpu().sum().float()
        correct += correct_idx
        acc.update(100. * correct_idx / float(targets.size(0)))
        train_loss_avg += loss.item()

        optimizer.zero_grad()
        loss.backward()
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

            mu_dis = 0
            gamma_dis = 0
            layer = 0
            lambda_mu = 0
            lambda_gamma = 0
            rho_mu = 0
            rho_gamma = 0
            if args.norm_layer == 'mv_aug':
                for m in net.modules():
                    if isinstance(m, BatchNorm_augmented2d):
                        mu_dis += m.mu_dis
                        gamma_dis += m.gamma_dis
                        lambda_mu += (m.alpha_**2).mean()
                        lambda_gamma += (m.beta_**2).mean()
                        rho_mu += (m.alpha1_**2).mean()
                        rho_gamma += (m.beta1_**2).mean()
                        layer+=1
                mu_dis/=float(layer)
                gamma_dis/=float(layer)
                lambda_mu /=float(layer)
                lambda_gamma /=float(layer)
                rho_mu /=float(layer)
                rho_gamma/=float(layer)
                logger.info("mu_dis: {:.3f}, gamma_dis: {:.3f} lambda mu {:.3f} gamma {:.3f} rho mu {:.3f} gamma {:.3f}".format(mu_dis, gamma_dis, lambda_mu, lambda_gamma, rho_mu, rho_gamma))

    tb_logger.add_scalar("train/train_loss_epoch", train_loss_avg / len(trainloader), epoch)
    tb_logger.add_scalar("train/train_acc_epoch", 100.*correct/total, epoch)
    wandb.log({"train/acc_epoch" : 100.*correct/total}, step=epoch)
    wandb.log({"train/loss_epoch" : train_loss_avg/len(trainloader)}, step=epoch)
    wandb.log({"train/mu_dis": mu_dis}, step=epoch)
    wandb.log({"train/gamma_dis": gamma_dis}, step=epoch)


    logger.info("epoch: {} acc: {}, loss: {}".format(epoch, 100.* correct/total, train_loss_avg / len(trainloader)))
    return (train_loss.avg, reg_loss.avg, 100.*correct/total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = AverageMeter(100)
    acc = AverageMeter(100)
    acc2 = AverageMeter(100)
    acc3 = AverageMeter(100)
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
            acc1, acc2_, acc3_ = accuracy(outputs, targets, topk=(1,2,3))

            correct_idx = predicted.eq(targets.data).sum().item()
            correct += correct_idx

            acc.update(100. * correct_idx / float(targets.size(0)))
            acc2.update(float(acc2_))
            acc3.update(float(acc3_))

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
    logger.info("acc2: {}".format(acc2.avg))
    logger.info("acc3: {}".format(acc3.avg))

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

from models.batchrenorm import BatchRenorm2d
from models.batchnorm import BatchNorm2d
if use_cuda:
    device=torch.device("cuda")
for m in net.modules():
    if isinstance(m, (BatchRenorm2d, BatchNorm2d)):
        m.sample_noise=args.sample_noise
        m.sample_mean = torch.ones(m.num_features).to(device)
        m.noise_std_mean=torch.sqrt(torch.Tensor([args.noise_std_mean]))[0].to(device)
        m.noise_std_var=torch.sqrt(torch.Tensor([args.noise_std_var]))[0].to(device)
if args.test == True:
    test(0)
for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    wandb.log({"test/train_test_loss_gap": test_loss - train_loss}, step=epoch)
    wandb.log({"test/train_test_acc_gap": train_acc - test_acc}, step=epoch)
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
