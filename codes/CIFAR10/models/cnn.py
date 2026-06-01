"""
Custom CNN for CIFAR-10 (Task 1).

Architecture: stacked conv blocks with BN, ReLU, MaxPool, and a FC classifier
with Dropout. Designed for 32x32 inputs without relying on pretrained backbones.
"""

import numpy as np
import torch
from torch import nn

from common.utils.nn import init_weights_kaiming_

ACTIVATIONS = {
    'relu': nn.ReLU,
    'leaky_relu': lambda: nn.LeakyReLU(0.1, inplace=True),
    'gelu': nn.GELU,
}


class ConvBlock(nn.Module):
    """Two 3x3 conv layers followed by max pooling."""

    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        act_cls = ACTIVATIONS[activation]
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            act_cls(inplace=True) if activation == 'relu' else act_cls(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            act_cls(inplace=True) if activation == 'relu' else act_cls(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class CIFARNet(nn.Module):
    """
    Custom CNN for CIFAR-10.

    32x32 -> [block x4] -> 2x2 feature map -> GAP -> FC head.

    Required components: Conv2d, BatchNorm, ReLU, MaxPool, Linear, Dropout.
    """

    def __init__(
        self,
        num_classes=10,
        width=64,
        dropout=0.5,
        activation='relu',
        init_weights=True,
    ):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, width, activation=activation),
            ConvBlock(width, width * 2, activation=activation),
            ConvBlock(width * 2, width * 4, activation=activation),
            ConvBlock(width * 4, width * 8, activation=activation),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 8, 256),
            ACTIVATIONS[activation](inplace=True) if activation == 'relu' else ACTIVATIONS[activation](),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        if init_weights:
            self.apply(init_weights_kaiming_)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def first_conv_weights(self):
        """Return first conv layer weights for visualization."""
        return self.features[0].block[0].weight.detach().cpu()


def count_parameters(model):
    return sum(np.prod(p.shape).item() for p in model.parameters())


if __name__ == '__main__':
    model = CIFARNet()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print('output shape:', y.shape)
    print('parameters:', count_parameters(model))
