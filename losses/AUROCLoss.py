
# %%
import torch
import torch.nn as nn
from abc import abstractmethod
import math
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score


def assertNan(x):
    try:
        assert torch.sum(torch.isnan(x)) == 0
    except:
        print(x)
        raise RuntimeError("asd")


class AUCLoss(nn.Module):

    '''Implementation of 
        "Zhiyong Yang, Qianqian Xu, Shilong Bao, Xiaochun Cao and Qingming Huang. 
            Learning with Multiclass AUC: Theory and Algorithms. T-PAMI, 2021."

        args:
            num_classes: number of classes (mush include params)

            gamma: safe margin in pairwise loss (default=1.0) 

            transform: manner to compute the multi-classes AUROC Metric, either 'ovo' or 'ova' (default as 'ovo' in our paper)

    '''

    def __init__(self,
                 num_classes,
                 gamma=1,
                 transform='ovo', *kwargs):
        super(AUCLoss, self).__init__()

        if transform != 'ovo' and transform != 'ova':
            raise Exception("type should be either ova or ovo")
        self.num_classes = num_classes
        self.gamma = gamma
        self.transform = transform

        if kwargs is not None:
            self.__dict__.update(kwargs)

    def forward(self, pred, target):
        mask = torch.stack(
            [target.eq(i).float() for i in range(self.num_classes)],
            1).squeeze()  # mask

        N = mask.sum(0)  # [类1的样本数目, ...]

        D = 1 / N[target.squeeze().long()]
        self.exist_class_num = torch.sum(N != 0)
        if self.transform == 'ovo':
            factor = self.exist_class_num * (self.exist_class_num - 1)
        else:
            factor = 1

        loss = torch.Tensor([0.]).cuda()
        if self.transform == 'ova':
            ones_vec = torch.ones_like(D).cuda()

        for i in range(self.num_classes):
            if N[i] == 0:
                continue
            if self.transform == 'ovo':
                Di = D / N[i]  # [1 / (n_i x n_1), dots, 1 / (n_i x n_c)]
            else:
                fac = torch.tensor([1.0]).cuda() / (N[i] * (N.sum() - N[i]))
                Di = fac * ones_vec
            mask_i, predi = mask[:, i], pred[:, i]

            asdasd = self.calLossPerCLass(predi, mask_i, Di, N[i])
            assertNan(asdasd)
            loss += asdasd

        return loss / factor

    def calLossPerCLass(self, predi, Yi, Di, Ni):

        return self.calLossPerCLassNaive(predi, Yi, Di, Ni)

    @abstractmethod
    def calLossPerCLassNaive(self, predi, Yi, Di, Ni):
        pass


class SquareAUCLoss(AUCLoss):
    def __init__(self, num_classes, gamma=1, transform='ovo', **kwargs):
        super(SquareAUCLoss, self).__init__(num_classes, gamma, transform)

        # self.num_classes = num_classes
        # self.gamma = gamma

        if kwargs is not None:
            self.__dict__.update(kwargs)

    def calLossPerCLassNaive(self, predi, Yi, Di, Ni):
        diff = predi - Yi
        nD = Di.mul(1 - Yi)
        fac = self.exist_class_num - 1
        S = Ni * nD + (fac * Yi / Ni)
        A = diff.mul(S).dot(diff)
        B = diff.dot(nD) * Yi.dot(diff)
        return 0.5 * A - B

# 不work，不清楚为啥
# class HingeAUCLoss(AUCLoss):
#     def __init__(self, num_classes, gamma=1, transform='ovo', **kwargs):
#         super(HingeAUCLoss, self).__init__(num_classes, gamma, transform)

#         if kwargs is not None:
#             self.__dict__.update(kwargs)

#     def calLossPerCLassNaive(self, predi, Yi, Di, Ni):
#         fac = 1 if self.transform == 'ova' else (self.num_classes - 1)
#         delta1 = (fac / Ni) * Yi * predi
#         delta2 = Di * (1 - Yi) * predi
#         return fac - delta1.sum() + delta2.sum()

# 自己重写一个
class HingeAUCLoss(nn.Module):

    def __init__(self, num_classes):
        super(HingeAUCLoss, self).__init__()
        self.num_classes = num_classes

    def hingeLoss(self, x):
        if x.is_cuda:
            return torch.max(1-x, torch.tensor(0).cuda())
        return torch.max(1-x, torch.tensor(0))

    def forward(self, pred, target):
        # pred : batch_size x class_num
        # target : batch_size

        mask = torch.stack(
            [target.eq(i) for i in range(self.num_classes)],
            1).squeeze()  # mask
        N = mask.sum(0)  # [类1的样本数目, ...]

        self.exist_class_num = torch.sum(N != 0)

        metric = 0
        loss = torch.Tensor([0.]).cuda()

        for i in range(self.num_classes):
            if N[i] == 0:
                continue
            mask_i = mask[:, i]

            f_i = pred[mask_i, :][:, i]
            f_not_i = pred[~mask_i, :][:, i]

            i_sample_num = f_i.shape[0]
            not_i_sample_num = f_not_i.shape[0]
            matrix_size = torch.Size((not_i_sample_num, i_sample_num))

            f_i = f_i.unsqueeze(0).expand(matrix_size)
            f_not_i = f_not_i.unsqueeze(1).expand(matrix_size)

            margin = (f_i - f_not_i)
            N_j = N[target[~mask_i]].unsqueeze(1).expand(matrix_size)
            metric += torch.sum(((margin > 0).float() + (margin == 0).float()/2)  / N_j / N[i])
            loss += torch.sum(self.hingeLoss(margin)  / N_j / N[i])

        metric = metric / self.exist_class_num / (self.exist_class_num - 1)
        loss = loss / self.exist_class_num / (self.exist_class_num - 1)
        return loss

class ExpAUCLoss(AUCLoss):
    def __init__(self, num_classes, gamma=1, transform='ovo', **kwargs):
        super(ExpAUCLoss, self).__init__(num_classes, gamma, transform)

        if kwargs is not None:
            self.__dict__.update(kwargs)

    def calLossPerCLassNaive(self, predi, Yi, Di, Ni):
        C1 = Yi * torch.exp(-self.gamma * predi)
        C2 = (1 - Yi) * torch.exp(self.gamma * predi)
        C2 = Di * C2
        return C1.sum() * C2.sum()

class AUCLoss_1(nn.Module):

    def __init__(self,
                 num_classes,
                 margin=None,
                 gamma=1,
                 transform='ovo', *kwargs):
        super(AUCLoss_1, self).__init__()

        if transform != 'ovo' and transform != 'ova':
            raise Exception("type should be either ova or ovo")
        self.num_classes = num_classes
        self.gamma = gamma
        self.transform = transform
        self.margin = margin
        if kwargs is not None:
            self.__dict__.update(kwargs)

    def forward(self, pred, target, cos):
        # pred : batch_size x class_num
        # target : batch_size

        mask = torch.stack(
            [target.eq(i) for i in range(self.num_classes)],
            1).squeeze()  # mask
        N = mask.sum(0)  # [类1的样本数目, ...]

        self.exist_class_num = torch.sum(N != 0)

        metric = 0
        loss_1 = torch.Tensor([0.]).cuda()
        loss_2 = torch.Tensor([0.]).cuda()

        for i in range(self.num_classes):
            if N[i] == 0:
                continue
            mask_i = mask[:, i]

            f_i = pred[mask_i, :]
            f_not_i = pred[~mask_i, :]

            i_sample_num = f_i.shape[0]
            not_i_sample_num = f_not_i.shape[0]
            feature_dim = f_i.shape[-1]

            matrix_size = torch.Size(
                (not_i_sample_num, i_sample_num, feature_dim))

            f_a = f_i.unsqueeze(0).expand(matrix_size).reshape(-1, feature_dim)
            f_b = f_not_i.unsqueeze(1).expand(
                matrix_size).reshape(-1, feature_dim)

            i_index = torch.ones(
                size=[not_i_sample_num * i_sample_num]).long() * i

            if self.margin is not None:
                margin = cos(i_index, f_a - f_b) - self.margin[i]
            else:
                margin = cos(i_index, f_a - f_b)
            N_j = N[target[~mask_i]].unsqueeze(1).expand(
                torch.Size([not_i_sample_num, i_sample_num])).reshape(-1)
            loss_1 += torch.sum(((1 - margin)**2 / N_j /
                                N[i])) / (self.exist_class_num - 1)

            margin = cos(i_index, (f_a - f_b).detach())
            margin_mean = torch.sum(
                margin / N_j / N[i]) / (self.exist_class_num - 1)
            margin_var = torch.sum(
                (margin - margin_mean) ** 2 / N_j / N[i]) / (self.exist_class_num - 1)
            loss_2 += torch.abs(margin_var / margin_mean)

        loss_1 /= self.exist_class_num
        loss_2 /= self.exist_class_num

        des = {
            # "AUC_mu": metric.item()
        }

        return loss_1, loss_2, des

class AUC_mu(nn.Module):

    def __init__(self,
                 num_classes,
                 surrogate,
                 gamma=1,
                 a=1,
                 transform='ovo', *kwargs):
        super(AUC_mu, self).__init__()

        if transform != 'ovo' and transform != 'ova':
            raise Exception("type should be either ova or ovo")
        self.num_classes = num_classes
        self.gamma = gamma
        self.a = a
        self.transform = transform
        self.surrogate = surrogate
        self.surrogate_loss = {
            "square": self.squareLoss,
            "exp": self.expLoss,
            "hinge": self.hingeLoss,
        }[self.surrogate]

        if kwargs is not None:
            self.__dict__.update(kwargs)

    def squareLoss(self, x):
        return (self.a-x)**2

    def hingeLoss(self, x):
        if x.is_cuda:
            return torch.max(1-x, torch.tensor(0).cuda())
        return torch.max(self.a-x, torch.tensor(0))

    def expLoss(self, x):
        return torch.exp(- self.a * x)

    def forward(self, pred, target):
        # pred : batch_size x class_num
        # target : batch_size

        mask = torch.stack(
            [target.eq(i) for i in range(self.num_classes)],
            1).squeeze()  # mask
        N = mask.sum(0)  # [类1的样本数目, ...]

        self.exist_class_num = torch.sum(N != 0)

        metric = 0
        loss = torch.Tensor([0.]).cuda()

        for i in range(self.num_classes):
            if N[i] == 0:
                continue
            mask_i = mask[:, i]

            f_i = pred[mask_i, :][:, i]
            f_not_i = pred[~mask_i, :][:, i]

            i_sample_num = f_i.shape[0]
            not_i_sample_num = f_not_i.shape[0]
            matrix_size = torch.Size((not_i_sample_num, i_sample_num))

            f_i = f_i.unsqueeze(0).expand(matrix_size)
            f_not_i = f_not_i.unsqueeze(1).expand(matrix_size)

            f_j_b = pred[~mask_i, target[~mask_i]]
            f_j_b = f_j_b.unsqueeze(1).expand(matrix_size)

            f_a = pred[mask_i, :]
            f_j_a = f_a[
                torch.arange(i_sample_num).cuda().unsqueeze(
                    0).expand(matrix_size),
                target[~mask_i].unsqueeze(1).expand(matrix_size)
            ]

            margin = (f_i - f_not_i + f_j_b - f_j_a)
            N_j = N[target[~mask_i]].unsqueeze(1).expand(matrix_size)
            metric += torch.sum(((margin > 0).float() + (margin == 0).float()/2)  / N_j / N[i])
            loss += torch.sum(self.surrogate_loss(margin)  / N_j / N[i])

        metric = metric / self.exist_class_num / (self.exist_class_num - 1)
        loss = loss / self.exist_class_num / (self.exist_class_num - 1)

        des = {
            "AUC_mu": metric.item()
        }

        return loss, des

class SoftF1(nn.Module):
    def __init__(self, num_classes):
        super(SoftF1, self).__init__()
        self.num_classes = num_classes

    def forward(self, pred, target):
        # pred : batch_size x class_num
        # target : batch_size
        pred_prob = torch.softmax(10 * pred, dim=1)
        soft_confusion_matrix = torch.zeros(self.num_classes, self.num_classes)

        for i in range(self.num_classes):
            mask_i = target==i
            soft_confusion_matrix[i, :] = torch.sum(pred_prob[mask_i], dim=0)

        recall = torch.zeros(self.num_classes)
        precsion = torch.zeros(self.num_classes)

        for i in range(self.num_classes):
            recall[i] = soft_confusion_matrix[i, i] / torch.sum(soft_confusion_matrix[i, :])
            precsion[i] = soft_confusion_matrix[i, i] / torch.sum(soft_confusion_matrix[:, i])

        res_numpy =  torch.argmax(pred, dim=1).detach().cpu().numpy()
        target_numpy =  target.detach().cpu().numpy()
        macro_f1 = f1_score(target_numpy, res_numpy, average="macro")

        soft_macro_f1 = torch.mean(2 * precsion * recall / (precsion + recall))
        return soft_macro_f1, macro_f1


    def forward(self, pred, target):
        # pred : batch_size x class_num
        # target : batch_size

        pred_prob = torch.softmax(10 * pred, dim=1)
        soft_confusion_matrix = torch.zeros(self.num_classes, self.num_classes)

        for i in range(self.num_classes):
            mask_i = target==i
            soft_confusion_matrix[i, :] = torch.sum(pred_prob[mask_i], dim=0)

        recall = []
        precsion = []

        for i in range(self.num_classes):
            recall.append(soft_confusion_matrix[i, i] / torch.sum(soft_confusion_matrix[i, :]))
            precsion.append(soft_confusion_matrix[i, i] / torch.sum(soft_confusion_matrix[:, i]))
        # print("recall", recall)
        # print("precsion", precsion)

        res_numpy =  torch.argmax(pred, dim=1).detach().cpu().numpy()
        target_numpy =  target.detach().cpu().numpy()
        macro_f1 = f1_score(target_numpy, res_numpy, average="macro")

        soft_macro_f1 = 0
        idx = 0
        for i in range(self.num_classes):
            aaaa = 2 * precsion[i] * recall[i] / (precsion[i] + recall[i])
            if not torch.isnan(aaaa):
                soft_macro_f1 += aaaa
                idx += 1

        return soft_macro_f1 / idx, macro_f1

class SquareAUCLoss_mine(nn.Module):

    def __init__(self,
                 num_classes,
                 gamma=1,
                 transform='ovo', *kwargs):
        super(SquareAUCLoss_mine, self).__init__()

        if transform != 'ovo' and transform != 'ova':
            raise Exception("type should be either ova or ovo")
        self.num_classes = num_classes
        self.gamma = gamma
        self.transform = transform

        if kwargs is not None:
            self.__dict__.update(kwargs)

    def forward(self, pred, target):
        # pred : batch_size x class_num
        # target : batch_size

        mask = torch.stack(
            [target.eq(i) for i in range(self.num_classes)],
            1).squeeze()  # mask
        N = mask.sum(0)  # [类1的样本数目, ...]

        self.exist_class_num = torch.sum(N != 0)

        metric = 0
        loss = torch.Tensor([0.]).cuda()

        for i in range(self.num_classes):
            if N[i] == 0:
                continue
            mask_i = mask[:, i]

            f_i = pred[mask_i, :][:, i]
            f_not_i = pred[~mask_i, :][:, i]

            i_sample_num = f_i.shape[0]
            not_i_sample_num = f_not_i.shape[0]

            matrix_size = torch.Size((not_i_sample_num, i_sample_num))

            f_i = f_i.unsqueeze(0)
            f_not_i = f_not_i.unsqueeze(1)
            margin = (f_i - f_not_i)
            N_j = N[target[~mask_i]].unsqueeze(1).expand(matrix_size)
            metric += torch.sum((margin > 0).float() / N_j /
                                N[i]) / (self.exist_class_num - 1)
            loss += torch.sum(((1 - margin)**2 / N_j /
                              N[i])) / (self.exist_class_num - 1)

        metric /= self.exist_class_num
        loss /= self.exist_class_num

        des = {
            "AUC": metric.item()
        }
        return loss

def focal_loss(logits, labels, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    
    # try:
    #     assertNan(logits)
    # except:
    #     print(logits)
    #     raise RuntimeError("asd")

    labels_onehot = torch.zeros_like(logits).float()
    labels_onehot[torch.arange(labels.size(0)), labels] = 1
    labels = labels_onehot
    logits = torch.sigmoid(logits)
    BCLoss = F.binary_cross_entropy(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    assertNan(modulator)

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss

def CB_loss(logits, labels, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes
    labels_one_hot = F.one_hot(labels, no_of_classes).float()
    weights = torch.tensor(weights).float()
    if logits.is_cuda:
        weights = weights.cuda()

    if loss_type == "focal":
        weights = weights.unsqueeze(0)
        weights = weights.expand_as(logits)
        cb_loss = focal_loss(logits, labels, weights, gamma)
    elif loss_type == "softmax":
        cb_loss = F.cross_entropy(input=logits, target=labels, weight=weights)
    return cb_loss

def balanced_softmax_loss(logits, labels, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      logits: A float tensor of size [batch, no_of_classes].
      labels: A int tensor of size [batch].
      sample_per_class: A list of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = torch.Tensor(sample_per_class).float().cuda()
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss

class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=10):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.from_numpy(m_list).float().cuda()
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(
            self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(output, target, weight=self.weight)

class LogitAdjustment2Loss(nn.Module):
    def __init__(self, cls_num_list):
        super(LogitAdjustment2Loss, self).__init__()
        class_num = len(cls_num_list)
        self.class_num = class_num
        self.cls_num_list = torch.Tensor(cls_num_list).cuda()
        self.pi_y = self.cls_num_list / torch.sum(self.cls_num_list)
        self.bias = 1
        self.weight = self.bias / self.pi_y
        self.factor = torch.log(1 / self.pi_y)

    # def forward(self, logits, labels):
    #     return F.cross_entropy(logits + self.bias, labels, weight=self.weight)

    def forward(self, logits, labels):
        batch_size = labels.shape[0]
        one_hot = torch.zeros(batch_size, self.class_num).float().cuda()
        one_hot[torch.arange(batch_size).cuda(), labels] = True

        factor = (1 / torch.sum(one_hot, dim=0))[labels]
        loss = torch.sum(factor * F.cross_entropy(logits * self.factor.unsqueeze(0), labels, reduction="none")) / torch.unique(labels).shape[0]
        return loss


class MarginCalibrationLoss(nn.Module):

    def __init__(self, cls_num_list):
        super(MarginCalibrationLoss, self).__init__()
        self.cls_num_list = cls_num_list

    def forward(self, x, target):
        return 0


# # %%
# if __name__ == "__main__":
#     batch_size = 200
#     class_num = 10
#     loss = AUC_mu(num_classes=10, surrogate="square")
#     preds = torch.randn(batch_size, class_num)
#     labels = torch.empty(batch_size, dtype=torch.long).random_(1, class_num)

#     # %%
#     def random_shuffle(preds, labels):
#         index = torch.randperm(batch_size)
#         preds = preds[index]
#         labels = labels[index]
#         return preds, labels
#     l, m = loss(preds, labels)
#     print(m)
#     preds_, labels_ = random_shuffle(preds, labels)
#     l, m = loss(preds, labels)
#     print(m)
