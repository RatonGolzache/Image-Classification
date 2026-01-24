import torch
import torch.nn as nn
import torch.nn.functional as F
import math
    
class Fire(nn.Module):
    """Fire module from SqueezeNet 1.1."""
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super().__init__()
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNetCIFAR10(nn.Module):
    """SqueezeNet 1.1 for CIFAR-10 (adapted from https://arxiv.org/abs/1602.07360)."""
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),  # 32x16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 32x7x7

            Fire(32,  16, 64,  64),  # 128x7x7
            Fire(128, 16, 64,  64),  # 128x7x7
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 128x3x3

            Fire(128, 32, 128, 128),  # 256x3x3
            Fire(256, 32, 128, 128),  # 256x3x3

            nn.Dropout(p=dropout),
            nn.Conv2d(256, num_classes, kernel_size=1),  # 10x1x1
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                    

class SqueezeNetFashionMNIST(nn.Module):
    """SqueezeNet 1.1 adapted for Fashion-MNIST (28x28 grayscale)."""
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            # Input: 1x28x28
            nn.Conv2d(1, 32, kernel_size=3, stride=2),  # 32x13x13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 32x5x5

            Fire(32, 16, 64, 64),  # 128x5x5
            Fire(128, 16, 64, 64),  # 128x5x5
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 128x2x2

            Fire(128, 32, 128, 128),  # 256x2x2
            nn.Dropout(p=dropout),
            nn.Conv2d(256, num_classes, kernel_size=1),  # 10x2x2
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()