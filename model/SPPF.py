import torch
from torch import nn
from CBS import CBS


class SPPF(nn.Module):
    """Implements a fast Spatial Pyramid Pooling (SPPF) layer for efficient feature extraction in YOLOv5 models."""

    def __init__(self, in_channels, out_channels):
        """
        Initializes YOLOv5 SPPF layer with given channels and kernel size for YOLOv5 model, combining convolution and
        max pooling.
        Equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = CBS(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv2 = CBS(hidden_channels * 4, out_channels, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))
