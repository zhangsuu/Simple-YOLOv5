from typing import Tuple
import torch
from torch import nn, Tensor
from C3 import C3
from CBS import CBS


class Neck(nn.Module):
    def __init__(self, depth_multiple: float = 1.0, width_multiple: float = 1.0):
        """

        """
        super().__init__()
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        self.cbs_k1_s1_c512 = CBS(in_channels=self.adjust_channels(1024), out_channels=self.adjust_channels(512),
                                  kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 颈部有两个c3_2_3_c512，上采样过程中使用的称为c3_2_3_c512_up，下采样过程中使用的称为c3_2_3_c512_down
        self.c3_2_3_c512_up = C3(self.adjust_channels(1024), self.adjust_channels(512), depth=self.adjust_depth(3),
                                 shortcut=False)
        self.cbs_k1_s1_c256 = CBS(in_channels=self.adjust_channels(512), out_channels=self.adjust_channels(256),
                                  kernel_size=1, stride=1)
        # 下采样
        self.c3_2_3_c256 = C3(self.adjust_channels(512), self.adjust_channels(256), depth=self.adjust_depth(3),
                              shortcut=False)
        self.cbs_k3_s2_c256 = CBS(in_channels=self.adjust_channels(256), out_channels=self.adjust_channels(256),
                                  kernel_size=3, stride=2)
        self.c3_2_3_c512_down = C3(self.adjust_channels(512), self.adjust_channels(512), depth=self.adjust_depth(3),
                                   shortcut=False)
        self.cbs_k3_s2_c512 = CBS(in_channels=self.adjust_channels(512), out_channels=self.adjust_channels(512),
                                  kernel_size=3, stride=2)
        self.c3_2_3_c1024 = C3(self.adjust_channels(1024), self.adjust_channels(1024), depth=self.adjust_depth(3),
                               shortcut=False)

    def forward(self, x: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        # 先上采样融合
        y1 = self.cbs_k1_s1_c512(x[2])
        y2 = self.cbs_k1_s1_c256(self.c3_2_3_c512_up(torch.cat([x[1], self.upsample(y1)], dim=1)))
        y3 = torch.cat([x[0], self.upsample(y2)], dim=1)
        # 下采样融合
        z1 = self.c3_2_3_c256(y3)
        z2 = self.c3_2_3_c512_down(torch.cat([self.cbs_k3_s2_c256(z1), y2], dim=1))
        z3 = self.c3_2_3_c1024(torch.cat([self.cbs_k3_s2_c512(z2), y1], dim=1))
        return z1, z2, z3

    def adjust_channels(self, channels):
        return max(int(channels * self.width_multiple), 1)

    def adjust_depth(self, depth):
        return max(int(depth * self.depth_multiple), 1)


if __name__ == '__main__':
    model = Neck()
    x1 = torch.randn(1, 256, 76, 76)
    x2 = torch.randn(1, 512, 38, 38)
    x3 = torch.randn(1, 1024, 19, 19)
    z1, z2, z3 = model((x1, x2, x3))
    print(z1.shape, z2.shape, z3.shape)
