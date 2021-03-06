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
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
from torch.utils.tensorboard import SummaryWriter
from utils import progress_bar, AverageMeter
from utils import create_logger
import yaml

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--load_model', type=str, default='')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=64, type=int, help='batch size')
parser.add_argument('--epoch', default=100, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--log_dir', default="oracle_exp001")
parser.add_argument('--grad_clip', default=3)
# for lr scheduler
parser.add_argument('--lr_ReduceLROnPlateau', default=False, type=bool)
parser.add_argument('--schedule', default=[60,80])
parser.add_argument('--fixup', default=False)
parser.add_argument('--decrease_affine', default=False)



# dataset
parser.add_argument('--dataset', default='CIFAR10', type=str)

parser.add_argument('--print_freq', default=10, type=int)


args = parser.parse_args()
with open("config/domain.yaml") as f:
    config = yaml.load(f)
for k, v in config['common'].items():
    setattr(args, k, v)

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

torch.manual_seed(args.seed)

os.makedirs('results/{}'.format(args.log_dir), exist_ok=True)
logger = create_logger('global_logger', "results/{}/log.txt".format(args.log_dir))

experiment = Experiment(api_key="1v0Sm8eioBxd9w0fhZq1FwE0g",
                        project_name="constraint_bn", workspace="lianqing11",
                        auto_output_logging=False,
                        log_env_gpu=False,
                        log_env_cpu=False,
                        log_env_host=False)
experiment.set_name(args.log_dir + time.asctime(time.localtime(time.time())).replace(" ", "-"))


experiment.add_tag('pytorch')
experiment.log_parameters(args.__dict__)
# Data
logger.info('==> Preparing data..')


from torch.utils.data import Dataset
import torchvision
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
class NormalDataset(Dataset):
    def __init__(self, root_dir, meta_file, is_train=True, args=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        print("building dataset from %s"%meta_file)
        self.metas = pd.read_csv(meta_file, sep=" ",header=None)
        print("read meta done")
        self.num = len(self.metas)
    def __len__(self):
        return self.num


    def __getitem__(self, idx):
        filename = osp.join(self.root_dir, self.metas.loc[idx, 0])

        label = self.metas.loc[idx, 1]
        ## memcached
        img = Image.open(filename).convert('RGB')
        #img = np.zeros((350, 350, 3), dtype=np.uint8)
        #img = Image.fromarray(img)
        #cls = 0

        ## transform
        if self.transform is not None:
            img = self.transform(img)
        return img, label




crop_size=224
val_size=256
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
train_source_dataset = NormalDataset(
    args.train_source_root,
    args.train_source_source,
    transform = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]), args=args )

train_target_dataset = NormalDataset(
    args.train_target_root,
    args.train_target_source,
    transform = transforms.Compose([
    transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]), args=args)

val_dataset = NormalDataset(
    args.val_root,
    args.val_source,
    transform = transforms.Compose([
        transforms.Resize(val_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ]),is_train=False, args=args )

source_val_dataset = NormalDataset(
    args.source_val_root,
    args.source_val_source,
    transform = transforms.Compose([
        transforms.Resize(val_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ]),is_train=False, args=args )


trainloader = DataLoader(
    train_source_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=4)

train_target_loader = DataLoader(
    train_target_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=4)

val_loader = DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=4)

source_val_loader = DataLoader(
    source_val_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=4)



# Model
if args.resume:
    # Load checkpoint.
    logger.info('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.load_model)
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
else:
    logger.info('==> Building model..')
    net = torchvision.models.__dict__[args.model](pretrained=True)
net.classifier[-1] = nn.Linear(4096, args.num_classes)

logname = ('results/{}/log_'.format(args.log_dir) + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')

tb_logger = SummaryWriter(log_dir="results/{}".format(args.log_dir))

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    logger.info(torch.cuda.device_count())
    cudnn.benchmark = True
    logger.info('Using CUDA..')

criterion = nn.CrossEntropyLoss()
logger.info(args.lr)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)

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
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        start = time.time()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

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
        optimizer.step()
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
            experiment.log_metric("loss_step", train_loss.avg, curr_idx)
            experiment.log_metric("acc_step", acc.avg, curr_idx)

    tb_logger.add_scalar("train/train_loss_epoch", train_loss_avg / len(trainloader), epoch)
    tb_logger.add_scalar("train/train_acc_epoch", 100.*correct/total, epoch)
    experiment.log_metric("acc_epoch", 100.*correct/total, epoch)
    experiment.log_metric("loss_epoch", train_loss_avg/len(trainloader), epoch)

    logger.info("epoch: {} acc: {}, loss: {}".format(epoch, 100.* correct/total, train_loss_avg / len(trainloader)))
    return (train_loss.avg, reg_loss.avg, 100.*correct/total)


def test(epoch, testloader, domain="source"):
    global best_acc
    net.eval()
    test_loss = AverageMeter(100)
    acc = AverageMeter(100)
    correct = 0
    total = 0
    logger.info("epoch: {} domain: {}".format(epoch, domain))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss.update(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct_idx = predicted.eq(targets.data).cpu().sum()
            correct += correct_idx

            acc.update(100. * correct_idx / float(targets.size(0)))
        progress_bar(batch_idx, len(testloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss.avg, acc.avg,
                        correct, total))
    acc = 100.*correct/total
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc
    tb_logger.add_scalar("test/test_loss_{}".format(domain), test_loss.avg, epoch * len(trainloader))
    tb_logger.add_scalar("test/test_acc_{}".format(domain), 100.*correct/total, epoch*len(trainloader))
    tb_logger.add_scalar("test/test_loss_epoch_{}".format(domain), test_loss.avg, epoch)
    tb_logger.add_scalar("test/test_acc_epoch_{}".format(domain), 100.*correct/total, epoch)
    experiment.log_metric("loss_step_{}".format(domain), test_loss.avg, epoch * len(trainloader))
    experiment.log_metric("acc_step_{}".format(domain), 100.*correct/total, epoch*len(trainloader))
    experiment.log_metric("loss_epoch_{}".format(domain), test_loss.avg, epoch)
    experiment.log_metric("acc_epoch_{}".format(domain), 100.*correct/total, epoch)

    return (test_loss.avg, 100.*correct/total)


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

lr_scheduler.step(start_epoch)
lr = optimizer.param_groups[0]['lr']
logger.info("epoch: {}, lr: {}".format(start_epoch, lr))

if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                            'test loss', 'test acc'])

for epoch in range(start_epoch, args.epoch):
    with experiment.train():
        train_loss, reg_loss, train_acc = train(epoch)
    with experiment.test():
        test_loss, test_acc = test(epoch, source_val_loader, domain="source")
        test_loss, test_acc = test(epoch, val_loader, domain="target")
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
