"""
Models for the loss landscape replication.

We use small networks (~20 layers) so the project is feasible on a laptop or
modest GPU. ResNet-20 and PlainNet-20 are paired architectures — identical
except for the skip connections — which is exactly what the original paper used
to isolate the effect of skip connections on the loss landscape.

Reference: He et al., "Deep Residual Learning for Image Recognition" (2016).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Residual block with optional skip connection."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_skip=True):
        super().__init__()
        self.use_skip = use_skip

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if use_skip:
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes * self.expansion:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes * self.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * self.expansion)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_skip:
            out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet for CIFAR-10. n=3 → ResNet-20, n=5 → ResNet-32, etc."""

    def __init__(self, n=3, num_classes=10, use_skip=True):
        super().__init__()
        self.in_planes = 16
        self.use_skip = use_skip

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, n, stride=1)
        self.layer2 = self._make_layer(32, n, stride=2)
        self.layer3 = self._make_layer(64, n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, self.use_skip))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(n=3, use_skip=True)


def plainnet20():
    return ResNet(n=3, use_skip=False)


if __name__ == "__main__":
    # Quick sanity check
    for name, model_fn in [("ResNet-20", resnet20), ("PlainNet-20", plainnet20)]:
        m = model_fn()
        n_params = sum(p.numel() for p in m.parameters())
        x = torch.randn(2, 3, 32, 32)
        y = m(x)
        print(f"{name}: {n_params:,} params, output shape {tuple(y.shape)}")
