
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PLLoss(nn.Module):
    def __init__(self, cls_num_list, loss_function):
        super(PLLoss , self).__init__()

    def forward(self, logits, label):
        return F.cross_entropy(logits, label)