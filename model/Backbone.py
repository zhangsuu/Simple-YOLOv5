from typing import Tuple

import torch
from torch import nn, Tensor

from C3 import C3
from CBS import CBS
from SPPF import SPPF


class Backbone(nn.Module):
    def __init__(self, in_channels: int = 3,
                 depth_multiple: float = 1.0,
                 width_multiple: float = 1.0):
        """

        """
        super().__init__()
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        self.cv1 = CBS(in_channels, self.adjust_channels(64), kernel_size=6, stride=2, padding=2)
        self.cv2 = CBS(self.adjust_channels(64), self.adjust_channels(128), kernel_size=3, stride=2)
        self.c3_1_3_c128 = C3(self.adjust_channels(128), self.adjust_channels(128), depth=self.adjust_depth(3),
                              shortcut=True)
        self.cv3 = CBS(self.adjust_channels(128), self.adjust_channels(256), kernel_size=3, stride=2)
        self.c3_1_6_c256 = C3(self.adjust_channels(256), self.adjust_channels(256), depth=self.adjust_depth(6),
                              shortcut=True)
        self.cv4 = CBS(self.adjust_channels(256), self.adjust_channels(512), kernel_size=3, stride=2)
        self.c3_1_9_c512 = C3(self.adjust_channels(512), self.adjust_channels(512), depth=self.adjust_depth(9),
                              shortcut=True)
        self.cv5 = CBS(self.adjust_channels(512), self.adjust_channels(1024), kernel_size=3, stride=2)
        self.c3_1_3_c1024 = C3(self.adjust_channels(1024), self.adjust_channels(1024), depth=self.adjust_depth(3),
                               shortcut=True)
        self.sppf = SPPF(self.adjust_channels(1024), self.adjust_channels(1024))

    def adjust_channels(self, channels):
        return max(int(channels * self.width_multiple), 1)

    def adjust_depth(self, depth):
        return max(int(depth * self.depth_multiple), 1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        y1 = self.c3_1_6_c256(self.cv3(self.c3_1_3_c128(self.cv2(self.cv1(x)))))
        y2 = self.c3_1_9_c512(self.cv4(y1))
        y3 = self.sppf(self.c3_1_3_c1024(self.cv5(y2)))
        return y1, y2, y3


if __name__ == '__main__':
    inputs = torch.randn(1, 3, 608, 608)
    backbone = Backbone()
    outputs = backbone(inputs)
    print(outputs[0].shape, outputs[1].shape, outputs[2].shape)
