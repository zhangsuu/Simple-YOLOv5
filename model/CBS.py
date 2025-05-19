from typing import Tuple, Union

import torch
from torch import nn


class CBS(nn.Module):
    """
    Applies a convolution, batch normalization, and activation function to an input tensor in a neural network.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = (1, 1),
                 padding: Union[int, Tuple[int, int]] = None,
                 dilation: int = 1,
                 groups: int = 1,
                 act: Union[bool, nn.Module] = True):
        """
        Initializes a standard convolution layer with optional batch normalization and activation.
        @param in_channels: Number of input channels.
        @param out_channels: Number of output channels.
        @param kernel_size: Kernel size.
        @param stride: Stride.
        @param padding: Padding.
        @param groups: Group size.
        @param dilation: Dilation.
        @param act: 是否使用激活函数，以及指定自定义的激活函数。
        """
        super().__init__()
        # 先统一参数样式
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if padding is None:
            padding = self.autopad(kernel_size, dilation)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              groups=groups, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if isinstance(act, nn.Module):
            self.act = act
        elif act is True:
            self.act = nn.SiLU()  # default activation
        elif act is False:
            self.act = None
        else:
            raise ValueError('invalid activation')

    def autopad(self, k: Tuple[int, int], d: int) -> Tuple[int, int]:
        """
        计算padding值，使卷积层输入和输出的特征图大小相同。
        @param k: 卷积核大小
        @param d: 卷积核的膨胀率, 用于空洞卷积
        @return: padding值
        """
        # 如果使用了空洞卷积，还要计算等效的卷积核的大小
        if d > 1:
            k = [d * (x - 1) + 1 for x in k]
        p = (k[0] // 2, k[1] // 2)
        return p

    def forward(self, x):
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Applies a fused convolution and activation function to the input tensor `x`."""
        return self.act(self.conv(x))


if __name__ == '__main__':
    inputs = torch.randn(1, 3, 416, 416)
    conv = CBS(3, 64, kernel_size=(3, 3))
    print(conv(inputs).shape)
