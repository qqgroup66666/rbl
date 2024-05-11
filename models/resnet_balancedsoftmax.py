# %%
"""
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import collections
from os import path


__all__ = [
    "ResNet",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnet1202",
]


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_Cifar(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet_Cifar, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.apply(_weights_init)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def load_model(self, pretrain):
        print("Loading Backbone pretrain model from {}......".format(pretrain))
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)
        pretrain_dict = pretrain_dict["state_dict"] if "state_dict" in pretrain_dict else pretrain_dict
        from collections import OrderedDict

        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                k = k[7:]
            if "last_linear" not in k and "classifier" not in k and "linear" not in k and "fd" not in k:
                k = k.replace("backbone.", "")
                k = k.replace("fr", "layer3.4")
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Backbone model has been loaded......")

    def forward(self, x, **kwargs):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        feature_maps = out
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out


class DotProduct_Classifier(nn.Module):
    
    def __init__(self, num_classes=1000, feat_dim=2048):
        super(DotProduct_Classifier, self).__init__()
        # print('<DotProductClassifier> contains bias: {}'.format(bias))
        self.fc = nn.Linear(feat_dim, num_classes)
        
    def forward(self, x):
        x = self.fc(x)
        return x


class ToyNet(nn.Module):
    def __init__(self):
        super(ToyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(12544, 64)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        return x

class _resnet8(nn.Module):
    def __init__(self, num_classes):
        super(_resnet8, self).__init__()
        self.encoder = ToyNet()
        # self.encoder = ResNet_Cifar(BasicBlock, [1, 1, 1])
        self.classifier = DotProduct_Classifier(num_classes, 64)

        self.model = nn.Sequential(collections.OrderedDict([
            ("encoder", self.encoder),
            ("classifier", self.classifier)
        ]))

    def forward(self, x):
        return self.classifier(self.encoder(x))

# class _resnet8_f2(nn.Module):
#     def __init__(self, num_classes):
#         super(_resnet8_f2, self).__init__()
#         print('Loading ResNet 32 Feature Model.')
#         self.encoder_1 = ToyNet()
#         self.encoder_2 = DotProduct_Classifier(2, 64)
#         self.classifier = DotProduct_Classifier(num_classes, 2)

#         self.model = nn.Sequential(collections.OrderedDict([
#             ("encoder", nn.Sequential(self.encoder_1, self.encoder_2)),
#             ("classifier", self.classifier)
#         ]))
#         self.max_norm = 10

#     def forward(self, x):
#         return self.classifier(
#             torch.nn.functional.normalize(self.encoder_2(self.encoder_1(x)), p=2, dim=1) * self.max_norm
#         )

#     def forward_feature(self, x):
#         return torch.nn.functional.normalize(self.encoder_2(self.encoder_1(x)), p=2, dim=1) * self.max_norm
    
#     def get_classweight(self):
#         return self.classifier.fc.weight.data

def _resnet32(num_classes, use_fc=False, pretrain=False, dropout=None, stage1_weights=False, dataset=None, log_dir=None,
                 test=False, *args):
    print('Loading ResNet 32 Feature Model.')
    encoder = ResNet_Cifar(BasicBlock, [5, 5, 5])
    classifier = DotProduct_Classifier(num_classes, 64)

    model = nn.Sequential(collections.OrderedDict([
        ("encoder", encoder),
        ("classifier", classifier)
    ]))
    return model


def _resnet50(num_classes, use_fc=False, pretrain=False, dropout=None, stage1_weights=False, dataset=None, log_dir=None,
                 test=False, *args):
    print('Loading ResNet 50 Feature Model.')
    encoder = ResNet_Cifar(BasicBlock, [9, 9, 9])
    classifier = DotProduct_Classifier(num_classes, 64)

    model = nn.Sequential(collections.OrderedDict([
        ("encoder", encoder),
        ("classifier", classifier)
    ]))
    return model
