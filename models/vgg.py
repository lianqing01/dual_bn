'''VGG11/13/16/19 in Pytorch.'''
import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from .dual_norm import DualNorm
from .MABN import MABN2d
from .dual_norm import DualAffine
from .constraint_bn_v2 import *
from .batchnorm_sample import BatchNorm_unbias2d
from .batchnorm import BatchNorm2d
from .instancenorm import InstanceNorm2d
from .batchrenorm import BatchRenorm2d
import torch.nn.utils.weight_norm as weightNorm

from .batch_renormalization import BatchRenormalization2D

class Conv_Cen2d(nn.Module):
    """Conv2d layer with Weight Centralization
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(Conv_Cen2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.randn(out_planes, in_planes//groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_planes))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)




cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG500': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256] + [256]*500 + ['M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG50': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256] + [256]*50 + ['M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG25': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256] + [256]*25 + ['M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],

}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes = 10, with_bn=False):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], with_bn)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        # Initialize weights
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
       '''

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, with_bn=False):
        layers = []
        in_channels = 3
        for idx, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if with_bn == 'dual': # deprecated
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               DualNorm(x),
                               nn.ReLU(inplace=True),
                               DualAffine(x),]
                elif with_bn == 'bn':
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                elif with_bn == 'ubbn':
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               BatchNorm_unbias2d(x),
                               nn.ReLU(inplace=True)]
                elif with_bn == 'bn_v2':
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.BatchNorm2d(x)]
                elif with_bn == 'bn_population':
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               BatchNorm2d(x),
                               nn.ReLU(inplace=True)]

                elif with_bn == 'brn':
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               BatchRenorm2d(x),
                               nn.ReLU(inplace=True)]

                elif with_bn == 'constraint_bn_v2':
                    if idx == 0:
                        layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True)]
                    else:
                        layers += [Constraint_Norm2d(in_channels, pre_affine=True, post_affine=True),
                                   #Constraint_Affine2d(in_channels),
                                   nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True)]
                elif with_bn == 'constraint_bn_v3':
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                                Constraint_Norm2d(x, pre_affine=True, post_affine=True),
                                nn.ReLU(inplace=True)]
                elif with_bn =='gn':
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                                nn.GroupNorm(32, x, affine=True),
                               nn.ReLU(inplace=True)]
                elif with_bn == 'constraint_bn_v2_no_affine':
                    if idx == 0:
                        layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True)]
                    else:
                        layers += [Constraint_Norm2d(in_channels, pre_affine=False, post_affine=False),
                                   #Constraint_Affine2d(in_channels),
                                   nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True)]

                elif with_bn == 'in':
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               InstanceNorm2d(x, affine=True),
                               nn.ReLU(inplace=True),]
                elif with_bn == 'mabn':
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               MABN2d(x),
                               nn.ReLU(inplace=True)]
                elif with_bn == 'wn':
                    layers += [weightNorm(nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=True), name="weight"),
                               nn.ReLU(inplace=True)]
                elif with_bn == 'mabn_cen':
                    layers += [Conv_Cen2d(in_channels, x, kernel_size=3, padding=1),
                               MABN2d(x),
                               nn.ReLU(inplace=True),]

                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def vgg16(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn=False)


def vgg16_mabn(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='mabn')


def vgg16_mabn_cen(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='mabn_cen')

def vgg16_wn(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='wn')





def vgg500(num_classes=10):
    return VGG('VGG500', num_classes=num_classes, with_bn=False)


def vgg50(num_classes=10):
    return VGG('VGG50', num_classes=num_classes, with_bn=False)
def vgg25(num_classes=10):
    return VGG('VGG25', num_classes=num_classes, with_bn=False)


def vgg500_bn(num_classes=10):
    return VGG('VGG500', num_classes=num_classes, with_bn='bn')
def vgg50_bn(num_classes=10):
    return VGG('VGG50', num_classes=num_classes, with_bn='bn')

def vgg50_constraint_bn_v2(num_classes=10):
    return VGG('VGG50', num_classes=num_classes, with_bn='constraint_bn_v2')

def vgg25_constraint_bn_v2(num_classes=10):
    return VGG('VGG25', num_classes=num_classes, with_bn='constraint_bn_v2')

def vgg16_bn(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='bn')

def vgg16_bn_v2(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='bn_v2')

def vgg16_bn_moving_average(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='bn_moving_average')
def vgg16_pn(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='bn_population')

def vgg16_in(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='in')

def vgg16_gn(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='gn')


def vgg16_brn(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='brn')


def vgg16_dual_bn(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='dual')

def vgg16_constraint_bn_v2(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='constraint_bn_v2')

def vgg16_constraint_bn_v3(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='constraint_bn_v3')

def vgg16_ubbn(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='ubbn')




def vgg16_constraint_bn_v2_noaffine(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='constraint_bn_v2_no_affine')
# net = VGG('VGG11')
# x = torch.randn(2,3,32,32)
# print(net(Variable(x)).size())
