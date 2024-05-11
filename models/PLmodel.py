import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import os
import geotorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

import random

# 标准方法
class PLPostHocModel(nn.Module):
    def __init__(self, backbone, triv, feature_num, class_num, _cls_num_list):
        super(PLPostHocModel, self).__init__()
        self.feature_num = feature_num
        self.class_num = class_num
        self.backbone = backbone
        _cls_num_list = torch.Tensor(_cls_num_list)
        self.margin = torch.log(_cls_num_list / torch.sum(_cls_num_list)).cuda()

        if feature_num < class_num:
            self.rotate = nn.Linear(class_num, feature_num, bias=False)
            self.register_buffer("EFT", self.generate_ETF(dim=class_num))
        else:
            self.rotate = nn.Linear(feature_num, feature_num, bias=False)
            self.register_buffer("EFT", \
                self.generate_ETF(dim=feature_num)[:, :self.class_num])
        geotorch.orthogonal(self.rotate, "weight", triv=triv)

    def generate_ETF(self, dim):
        return torch.eye(dim, dim) - torch.ones(dim, dim) / dim

    def forward(self, x):
        logit = self.backbone(x) @ self.rotate.weight @ self.EFT
        return logit if self.training else logit - self.margin

    def forward_feature(self, x):
        return self.backbone(x)

    def get_classweight(self):
        return (self.rotate.weight @ self.EFT).T

if __name__ == "__main__":
    
    x = torch.randn(5, 3, 32, 32).cuda()
    class_num = 10
    feature_num = 256
    backbone = resnet32(num_experts=2, num_classes=feature_num)
    model = PLModel(backbone, feature_num=feature_num, class_num=class_num).cuda()
    print(model(x).shape)
