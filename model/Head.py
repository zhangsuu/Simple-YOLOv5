from typing import Tuple
import torch
from torch import nn, Tensor
from C3 import C3
from CBS import CBS


class Head(nn.Module):
    def __init__(self, width_multiple: float = 1.0):
        super().__init__()
        self.width_multiple = width_multiple
        self.h1 = nn.Conv2d(in_channels=self.adjust_channels(1024), out_channels=255, kernel_size=1, stride=1)
        self.h2 = nn.Conv2d(in_channels=self.adjust_channels(512), out_channels=255, kernel_size=1, stride=1)
        self.h3 = nn.Conv2d(in_channels=self.adjust_channels(256), out_channels=255, kernel_size=1, stride=1)
    def adjust_channels(self, channels):
        return max(int(channels * self.width_multiple), 1)
    def forward(self, x: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        x1, x2, x3 = x
        x3 = self.h1(x3)
        x2 = self.h2(x2)
        x1 = self.h3(x1)
        return x1, x2, x3