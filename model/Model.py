from typing import Tuple
import torch
from torch import nn, Tensor
from model.Backbone import Backbone
from model.Head import Head
from model.Neck import Neck


class Model(nn.Module):
    def __init__(self, depth_multiple: float = 1.0, width_multiple: float = 1.0, *args, **kwargs):
        super().__init__()
        self.backbone = Backbone(depth_multiple=depth_multiple, width_multiple=width_multiple)
        self.neck = Neck(depth_multiple=depth_multiple, width_multiple=width_multiple)
        self.head = Head(width_multiple=width_multiple)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

if __name__ == '__main__':
    model = Model()
    print(model)
    # 计算参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 官方的YOLOv5l的参数量为46563709
    assert params == 46563709
    y1,  y2, y3 = model(torch.randn(1, 3, 608, 608))
    print(y1.shape, y2.shape, y3.shape)