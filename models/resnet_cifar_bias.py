import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


__all__ = ['resnet_bias', 'resnet_bias20', 'resnet_bias32', 'resnet_bias44', 'resnet_bias56', 'resnet_bias110', 'resnet_bias1202']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)



class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 resnet_bias paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     norm_layer(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class resnet_bias(nn.Module):

    def __init__(self, block, layers, num_classes=10, norm_layer=None):
        super(resnet_bias, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.num_layers = sum(layers)
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = norm_layer(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, norm_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride, norm_layer):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride, norm_layer=norm_layer))
            self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet_bias20(**kwargs):
    """Constructs a resnet_bias-20 model.

    """
    model = resnet_bias(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet_bias32(**kwargs):
    """Constructs a resnet_bias-32 model.

    """
    model = resnet_bias(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet_bias44(**kwargs):
    """Constructs a resnet_bias-44 model.

    """
    model = resnet_bias(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet_bias56(**kwargs):
    """Constructs a resnet_bias-56 model.

    """
    model = resnet_bias(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet_bias110(**kwargs):
    """Constructs a resnet_bias-110 model.

    """
    model = resnet_bias(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet_bias1202(**kwargs):
    """Constructs a resnet_bias-1202 model.

    """
    model = resnet_bias(BasicBlock, [200, 200, 200], **kwargs)
    return model
