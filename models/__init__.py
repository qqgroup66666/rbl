import torch.nn as nn
import torch.nn.functional as F
from .resnet_balancedsoftmax import _resnet8, _resnet32, _resnet50
from .resnext_sade import resnext50
from .sade_resnet import resnet50
from .mlp import mlp
from .PLmodel import PLPostHocModel
import torch
import math
import os
import geotorch

feature_dim_mapping = {
    "densenet121": 1024,
    "resnet18": 512,
    "resnet34": 512
}

class LinearClassifier(nn.Module):
    def __init__(self, feature_dim_num, num_classes):
        super(LinearClassifier, self).__init__()
        self.weight = nn.Parameter(torch.empty((feature_dim_num, num_classes), dtype=torch.float32))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # x : torch.size([batch_size, feature_dim])
        return torch.matmul(x, self.weight)

class MLP(nn.Module):
    """projection head"""
    def __init__(self, dim_in, dim_out, norm=True, need_feature=False):
        super(MLP, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.BatchNorm1d(dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, dim_out)
        )
        self.norm = norm
        self.need_feature = need_feature

    def set_need_feature(self, need_feature):
        self.need_feature = need_feature

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        if self.need_feature:
            feature = self.head[:-1](x)
            x = self.head[-1](feature)
        else:
            x = self.head(x)

        if self.norm:
            x = F.normalize(x, dim=1)

        if self.need_feature:
            return x, feature
        else:
            return x

class NormalizeFeature(nn.Module):
    """projection head"""
    def __init__(self):
        super(NormalizeFeature, self).__init__()

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return x

# 不同view的数据 进入 网络的不同分支
class BCLMultiBranchModel(nn.Module):
    """projection head"""
    def __init__(self, backbone, mlp_dim_in, mlp_dim_out, num_classes):
        super(BCLMultiBranchModel, self).__init__()
        self.backbone = backbone
        self.mlp = MLP(mlp_dim_in, mlp_dim_out)
        self.mlp_for_prototype = MLP(mlp_dim_in, mlp_dim_out)
        self.classifier = LinearClassifier(mlp_dim_in, num_classes)

    def forward(self, x):
        # 输入为3个view的图片数据
        # x: size([batch_size, 3, channel, w, h])
        x1 = self.backbone(x[:, 0, ::])
        x2 = self.backbone(x[:, 1, ::])
        x3 = self.backbone(x[:, 2, ::])
        return self.mlp(x1), self.mlp(x2), self.mlp_for_prototype(self.classifier.weight.T), self.classifier(x3)

    def get_inference_model(self):
        return BCLMultiBranchModel_inference_model(self)

class BCLMultiBranchModel_inference_model(nn.Module):
    def __init__(self, model):
        super(BCLMultiBranchModel_inference_model, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model.backbone(x)
        x = self.model.classifier(x)
        return x

# 不同view的数据 进入 网络的不同分支
class TCLModel(nn.Module):
    """projection head"""
    def __init__(self, query_encoder, key_encoder):
        super(TCLModel, self).__init__()
        self.query_encoder = query_encoder
        self.key_encoder = key_encoder

        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def forward(self, x):
        # 输入为3个view的图片数据
        # x: size([batch_size, 3, channel, w, h])
        query = self.query_encoder(x[:, 0, ::])
        key = self.key_encoder(x[:, 1, ::])
        return query, key

class ResLTModel(nn.Module):
    """projection head"""
    def __init__(self, backbone, feature_dim, num_classes):
        super(ResLTModel, self).__init__()
        self.backbone = backbone
        self.finalBlock = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim * 3, 1, bias=False),
                nn.BatchNorm2d(feature_dim * 3),
                nn.ReLU(inplace=True)
            )
        self.feature_dim = feature_dim
        self.fc = nn.Linear(feature_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.avgpool(x)
        x = self.finalBlock(x)
        x = x.reshape(batch_size, -1)
        many_x = x[:, :self.feature_dim]
        medium_x = x[:, self.feature_dim:2*self.feature_dim]
        few_x = x[:, 2*self.feature_dim:]

        if self.training:
            return self.fc(many_x), self.fc(medium_x), self.fc(few_x)
        else:
            return self.fc(many_x) + self.fc(medium_x) + self.fc(few_x)

class SADEModel(nn.Module):
    """projection head"""
    def __init__(self, backbone1, backbone2, backbone3, feature_dim, num_classes):
        super(SADEModel, self).__init__()
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.backbone3 = backbone3
        self.fc_1 = MLP(feature_dim, num_classes, norm=False)
        self.fc_2 = MLP(feature_dim, num_classes, norm=False)
        self.fc_3 = MLP(feature_dim, num_classes, norm=False)

    def forward(self, x):
        x_1 = self.fc_1(self.backbone1(x))
        x_2 = self.fc_2(self.backbone2(x))
        x_3 = self.fc_3(self.backbone3(x))
        if self.training:
            return torch.stack([x_1, x_2, x_3], dim=1)
        else:
            return x_1 + x_2 + x_3


class MarginCalibrationBackbone(nn.Module):
    """projection head"""
    def __init__(self, backbone, class_num):
        super(MarginCalibrationBackbone, self).__init__()
        self.backbone = backbone
        self.class_num = class_num
        self.omega = torch.ones(1, self.class_num).cuda()
        self.beta = torch.zeros(1, self.class_num).cuda()

    def forward(self, x):
        x = self.backbone(x)
        x = self.omega * x + self.beta * torch.norm(self.backbone.classifier.fc.weight, p=2, dim=1).unsqueeze(0)
        return x

class PostHocModel(nn.Module):
    """projection head"""
    def __init__(self, backbone, class_samples):
        super(PostHocModel, self).__init__()
        self.backbone = backbone
        self.margin = torch.log(torch.Tensor(class_samples) / sum(class_samples)).cuda()

    def forward(self, x):
        return self.backbone(x) if self.training else self.backbone(x) - self.margin

def generate_net(args, _cls_num_list):
    # if not args.model_type in globals().keys():
        # raise NotImplementedError("there has no %s" % (args.model_type))

    model_generator = globals()[args.model_type]
    model_use = args.get("model_use", "classifier")

    print("-" * 100)
    print("BackBone: ", args.model_type)
    print("model_use: ", model_use)
    print("-" * 100)

    if model_use == "feature_exactor": # 只需要特征提取器
        if args.model_type in ["resnet18", "resnet34", "_resnet32", "resnet50"]:
            model = model_generator(num_classes = args.feature_num)
        else:
            raise RuntimeError("?")
        feature_norm = args.get("feature_norm", False)
        if feature_norm:
            model = nn.Sequential(
                model,
                NormalizeFeature()
            )
    elif model_use == "classifier": # 分类
        if args.model_type in ["resnet18", "resnet34", "_resnet8_f2", "_resnet32", "resnet50", "_resnet50", "resnext50", "resnext50_norm"]:
            model = model_generator(num_classes = args.num_classes)
            pass
        else:
            raise RuntimeError("?")
    elif model_use == "BalancedSupConModel":
        if args.model_type in ["resnet18", "resnet34", "_resnet32", "resnet50"]:
            encoder = model_generator(num_classes = args.mlp_dim_in)
            model = BCLMultiBranchModel(
                backbone = encoder,
                mlp_dim_in = args.mlp_dim_in,
                mlp_dim_out = args.mlp_dim_out,
                num_classes = args.num_classes
            )
        else:
            raise RuntimeError("?")
    elif model_use == "TargetSupConModel":
        if args.model_type in ["resnet18", "resnet34", "_resnet32", "resnet50"]:
            key_encoder = model_generator(num_classes = args.feature_num)
            query_encoder = model_generator(num_classes = args.feature_num)
            model = TCLModel(
                query_encoder = query_encoder,
                key_encoder = key_encoder,
            )
        else:
            raise RuntimeError("?")
    elif model_use == "ResLTModel": 
        if args.model_type in ["resnet18", "resnet34", "_resnet32", "resnet50"]:
            encoder = model_generator(num_classes = args.feature_num).encoder_without_flatten
            model = ResLTModel(backbone=encoder, feature_dim=feature_dim_mapping["resnet18"], num_classes=args.num_classes)
        else:
            raise RuntimeError("?")
    elif model_use == "SADEModel": 
        backbone_1 = model_generator(num_classes=args.feature_num)
        backbone_2 = model_generator(num_classes=args.feature_num)
        backbone_3 = model_generator(num_classes=args.feature_num)
        model = SADEModel(backbone_1, backbone_2, backbone_3, feature_dim=args.feature_num, num_classes=args.num_classes)
    elif model_use == "FixedPermutation_Rotation":
        backbone = model_generator(num_classes = args.feature_num)
        triv = args.get("triv", "expm") # "cayley"
        print("pl backbone triv:", triv)
        model = FixedPermutation_Rotation(backbone=backbone, triv=triv, feature_num=args.feature_num, class_num=args.num_classes, _cls_num_list=_cls_num_list)
    elif model_use == "PLBackbone":
        backbone = model_generator(num_classes = args.feature_num)
        triv = args.get("triv", "expm") # "cayley"
        print("pl backbone triv:", triv)
        model = PLModel(backbone=backbone, triv=triv, feature_num=args.feature_num, class_num=args.num_classes)
    elif model_use == "MarginCalibrationBackbone":
        backbone = model_generator(num_classes = args.num_classes)
        model = MarginCalibrationBackbone(backbone=backbone, class_num=args.num_classes)
    elif model_use == "neural_collpase":
        feature_num = args.feature_num
        num_classes = args.num_classes
        backbone = model_generator(num_classes = feature_num)
        model = nn.Sequential(
            backbone, 
            MLP(dim_in=feature_num, dim_out=num_classes, norm=False)
        )
    elif model_use in ["PLPostHocModel", "Ablation_CE", "Ablation_LD", "Ablation_Fixed", "Ablation_RBL"]:
        backbone = model_generator(num_classes = args.feature_num)
        triv = args.get("triv", "expm") # "cayley"
        print("pl backbone triv:", triv)
        if model_use == "PLPostHocModel":
            model = PLPostHocModel(backbone=backbone, triv=triv, feature_num=args.feature_num, class_num=args.num_classes, _cls_num_list=_cls_num_list)
        if model_use == "Ablation_CE":
            model = Ablation_CE(backbone=backbone, triv=triv, feature_num=args.feature_num, class_num=args.num_classes, _cls_num_list=_cls_num_list)
        if model_use == "Ablation_LD":
            model = Ablation_LD(backbone=backbone, triv=triv, feature_num=args.feature_num, class_num=args.num_classes, _cls_num_list=_cls_num_list)
        if model_use == "Ablation_Fixed":
            model = Ablation_Fixed(backbone=backbone, triv=triv, feature_num=args.feature_num, class_num=args.num_classes, _cls_num_list=_cls_num_list)
        if model_use == "Ablation_RBL":
            model = Ablation_RBL(backbone=backbone, triv=triv, feature_num=args.feature_num, class_num=args.num_classes, _cls_num_list=_cls_num_list)
    else:
        raise RuntimeError("??")
    return model

__all__ = ['generate_net']
