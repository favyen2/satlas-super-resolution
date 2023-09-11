"""
Taken from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/archs/discriminator_arch.py
Authors: xinntao
"""
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class Dry1(nn.Module):
    def __init__(self, num_in_ch, num_feat=64):
        super(Dry1, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # downsample -> 64x64
            nn.Conv2d(num_feat, num_feat*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat*2, num_feat*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # downsample -> 32x32
            nn.Conv2d(num_feat*2, num_feat*4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat*4, num_feat*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # downsample -> 16x16
            nn.Conv2d(num_feat*4, num_feat*8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat*8, num_feat*8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # downsample -> 8x8
            nn.Conv2d(num_feat*8, num_feat*8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat*8, num_feat*8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # downsample -> 4x4
            nn.Conv2d(num_feat*8, num_feat*8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat*8, num_feat*8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # downsample -> 2x2
            nn.Conv2d(num_feat*8, num_feat*8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat*8, num_feat*8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # downsample -> 1x1
            nn.Conv2d(num_feat*8, num_feat*8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat*8, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.layers(x)
