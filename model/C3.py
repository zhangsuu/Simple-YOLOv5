import torch
from torch import nn

from Bottleneck import Bottleneck
from CBS import CBS


class C3(nn.Module):
    """Implements a CSP Bottleneck module with three convolutions for enhanced feature extraction in neural networks."""

    def __init__(self, in_channels: int, out_channels: int,
                 depth: int = 1,
                 shortcut=True, groups: int = 1,
                 e: float = 0.5):
        """
        @param depth : Bottleneck的数量
        @param shortcut : Bottleneck中是否使用残差连接，区分为C3_1和C3_2
        @param e :  通道缩放比例
        """
        super().__init__()
        hidden_channels = int(out_channels * e)
        self.cv1 = CBS(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv2 = CBS(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv3 = CBS(2 * hidden_channels, out_channels, kernel_size=1)  # optional act=FReLU(c2)
        self.res_unit = nn.Sequential(
            *(Bottleneck(hidden_channels, hidden_channels, shortcut, groups=groups, e=1.0) for _ in range(depth)))

    def forward(self, x):
        y1 = self.res_unit(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))
