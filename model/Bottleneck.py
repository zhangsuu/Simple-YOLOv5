from torch import nn

from CBS import CBS


class Bottleneck(nn.Module):
    """
    A bottleneck layer with optional shortcut and group convolution for efficient feature extraction.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 shortcut: bool = True,
                 groups: int = 1,
                 e: float = 0.5):
        """
        @param in_channels: Number of input channels.
        @param out_channels: Number of output channels.
        @param shortcut: Whether to include a shortcut connection.
        @param groups: Number of groups for group convolution.
        @param e: 中间层的模型层数缩减率
        """
        super().__init__()
        hidden_channels = int(out_channels * e)
        self.cv1 = CBS(in_channels, hidden_channels, kernel_size=(1, 1), stride=(1, 1))
        self.cv2 = CBS(hidden_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        if self.add:
            return x + self.cv2(self.cv1(x))
        else:
            return self.cv2(self.cv1(x))
