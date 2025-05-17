from .gsem import GradSemiMaskModule

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class WideBasic(nn.Module):
    def __init__(self, in_channels, channels, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class GSeMWideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, alpha, beta):
        super(GSeMWideResNet, self).__init__()
        self.in_channels = 64

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        nStages = [self.in_channels, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nStages[0])
        self.gsem_module = GradSemiMaskModule(nStages[0], alpha, beta)
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, channels, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, channels, dropout_rate, stride))
            self.in_channels = channels

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gsem_module(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, output_size=1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def GSeMWideResNet28x10(num_classes=10, alpha=6, beta=4):
    return GSeMWideResNet(28, 10, 0, num_classes, alpha=alpha, beta=beta)


def GSeMWideResNet28x20(num_classes=10, alpha=6, beta=4):
    return GSeMWideResNet(28, 20, 0.3, num_classes, alpha=alpha, beta=beta)


def GSeMWideResNet40x10(num_classes=10, alpha=6, beta=4):
    return GSeMWideResNet(40, 10, 0.3, num_classes, alpha=alpha, beta=beta)


def GSeMWideResNet40x14(num_classes=10, alpha=6, beta=4):
    return GSeMWideResNet(40, 14, 0.3, num_classes, alpha=alpha, beta=beta)


if __name__ == '__main__':
    net = GSeMWideResNet28x10(10)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
