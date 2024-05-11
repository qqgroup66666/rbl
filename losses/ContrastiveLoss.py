import torch
import torch.nn as nn
import math
from .AUROCLoss import balanced_softmax_loss

def filter_only_one_sample_class(features, labels, class_num):
    # features: [batch_size, 1, feature_dim]
    # labels: [batch_size]
    # 用于剔除同类别只有他自己, 这样的样本. 这样的存在会造成Nan
    one_hot = torch.stack([labels==i for i in range(class_num)], dim=0)
    sample_num_class_in_batch = torch.sum(one_hot, dim=1).cuda()
    class_mask_for_sample_larger_than_1 = sample_num_class_in_batch > 1
    filter_mask = class_mask_for_sample_larger_than_1[labels] == True
    labels = labels[filter_mask]
    features = features[filter_mask, ::]
    return features, labels


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, class_num, temperature=0.1, base_temperature=0.07, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.class_num = class_num

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        if len(features.shape) < 3:
            features = features.unsqueeze(1)
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        # logits_mask 对角线为0, 其他全1， 为了剔除自己和自己对比
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

# 分母部分不再包含正例对
class SupConLoss_v2(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.1, base_temperature=0.07, contrast_mode='all'):
        super(SupConLoss_v2, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            view_num = int(features.shape[0] / labels.shape[0])
            split_ = torch.split(features, view_num, dim=0)
            split_ = [i.unsqueeze(0) for i in split_]
            features = torch.cat(split_, dim=0)
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        # logits_mask 对角线为0, 其他全1， 为了剔除自己和自己对比
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask__ = 1 - mask # 相较于原版的主要变化，分母部分不再包含正例对
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * mask__
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-4)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

class BalancedSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, sample_per_class, lambda_, mu_, temperature=0.1, contrast_mode='all'):
        super(BalancedSupConLoss , self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.lambda_ = lambda_ # ce loss
        self.mu_ = mu_ # sc loss
        self.sample_per_class = torch.tensor(sample_per_class).cuda()
        self.class_num = len(sample_per_class)

    def forward(self, feature1, feature2, prototype, logits, labels):
        # feature1: torch.size([batch_size, feature_dim])
        # feature2: torch.size([batch_size, feature_dim])
        # prototype: torch.size([class_num, feature_dim])
        # logits: torch.size([batch_size, class_num])
        # labels: torch.size([batch_size,])

        # print(feature1.shape)
        # print(feature2.shape)
        # print(prototype.shape)
        # print(logits.shape)
        # print(labels.shape)
        sc_loss = self.contrastive_loss(feature1, feature2, prototype, labels)
        ce_loss = balanced_softmax_loss(logits, labels, self.sample_per_class)
        return ce_loss, sc_loss, ce_loss * self.lambda_ + sc_loss * self.mu_

    def contrastive_loss(self, feature1, feature2, prototype, labels):
        device = (torch.device('cuda') if feature1.is_cuda else torch.device('cpu'))
        batch_size = labels.shape[0]

        # features = torch.cat([feature1, feature2], dim=0) 
        # labels = torch.cat([labels, labels], dim=0) # size([batch_size * 2 + class_num])

        class_num = prototype.shape[0]
        features = torch.cat([feature1, feature2, prototype], dim=0) # size([batch_size * 2 + class_num, feature_dim])
        class_index = torch.arange(class_num).to(device)
        labels = torch.cat([labels, labels, class_index], dim=0) # size([batch_size * 2 + class_num])

        # features, labels = filter_only_one_sample_class(features, labels, class_num=class_num)

        labels_ = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels_[:2 * batch_size], labels_.T).float().to(device)
        # logits_mask 对角线为0, 其他全1， 为了剔除自己和自己对比
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(2 * batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute logits
        sim = torch.div(
            torch.matmul(features[:2 * batch_size], features.T),
            self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        batch_cls_count = torch.eye(self.class_num).cuda()[labels].sum(dim=0).to(device)
        # batch_cls_count = torch.eye(self.class_num)[labels].sum(dim=0)
        # print(batch_cls_count.shape)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        exp_logits /= batch_cls_count[labels].unsqueeze(0) # 使用batch上的标签分布进行balance

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos.mean()
        return loss

class MultiSimilarityLoss(nn.Module):

    def __init__(self, scale_pos=2.0, scale_neg=40.0):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1
        self.scale_pos = scale_pos
        self.scale_neg = scale_neg

    def multi_view_to_single_view(self, feats, labels):
        labels = torch.cat([labels] * feats.shape[1], dim=0)
        feats = torch.split(feats, 1, dim=1)
        feats = [feat.squeeze(1) for feat in feats]
        feats = torch.cat(feats, dim=0)
        return feats, labels

    def forward_original(self, feats, labels):
    # def forward_(self, feats, labels):
        # 原版循环实现，很慢
        '''
            feats: tensor [batch_size, number of view, features dimension]
            labels: tensor [batch_size]
        '''

        batch_size = labels.shape[0]
        number_view = int(feats.shape[0] / batch_size)
        labels = labels.unsqueeze(0).expand(number_view, batch_size)
        labels = labels.reshape(-1)

        # 归一化
        if not (1 - 1e-5 < torch.sum(feats[0] * feats[0]) < 1 + 1e-5):
            feats = nn.functional.normalize(feats, p=2, dim=1)

        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            # Pair mining
            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]


            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # Pair weighting
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss

    def forward(self, feats, labels):
        '''
            feats: tensor [batch_size * number of view, features dimension]
            labels: tensor [batch_size]
        '''
        # if len(feats.shape) == 3:
        #     feats, labels = self.multi_view_to_single_view(feats, labels)
        # feats: tensor [batch_size, features dimension]
        # labels: tensor [batch_size]

        batch_size = labels.shape[0]
        number_view = int(feats.shape[0] / batch_size)
        labels = labels.unsqueeze(0).expand(number_view, batch_size)
        labels = labels.reshape(-1)

        # 归一化
        if not (1 - 1e-5 < torch.sum(feats[0] * feats[0]) < 1 + 1e-5):
            feats = nn.functional.normalize(feats, p=2, dim=1)

        sim = torch.matmul(feats, torch.t(feats))

        pos_mask = labels.unsqueeze(0).expand((labels.shape[0], labels.shape[0])) == \
            labels.unsqueeze(1).expand((labels.shape[0], labels.shape[0]))
        pos_mask[torch.eye(labels.shape[0], labels.shape[0]).bool()] = False
        neg_mask = labels.unsqueeze(0).expand((labels.shape[0], labels.shape[0])) != \
            labels.unsqueeze(1).expand((labels.shape[0], labels.shape[0]))

        # pair mining
        sim_ = sim.clone()
        sim_[~pos_mask] = 9999999
        min_value_pos_pair, _ = torch.min(sim_, dim=1)

        sim_ = sim.clone()
        sim_[~neg_mask] = -9999999
        max_value_neg_pair, _ = torch.max(sim_, dim=1)
        
        neg_mask_ = sim + self.margin > min_value_pos_pair.unsqueeze(1)
        pos_mask_ = sim < max_value_neg_pair.unsqueeze(1) + self.margin

        neg_mask = (neg_mask_ & neg_mask).float().detach()
        pos_mask = (pos_mask_ & pos_mask).float().detach()

        loss = torch.mean(
            torch.log(
                1 + torch.sum(torch.exp(- self.scale_pos * (sim - self.thresh)) * pos_mask, dim=1)
            ) / self.scale_pos + torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (sim - self.thresh)) * neg_mask, dim=1)
            ) / self.scale_neg
        )

        return loss

class TargetSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07):
        super(TargetSupConLoss , self).__init__()
        self.temperature = temperature
        
        self.optimal_target = None
        self.optimal_target = None

    def forward(self, query, key):
        # query: tensor, torch.size([batch_size, feature_dim])
        # key: tensor, torch.size([batch_size, feature_dim])
        print(query.shape) 
        print(key.shape)
        exit(0)
        loss = 0
        return loss

    def contrastive_loss(self, feature1, feature2, prototype, labels):
        device = (torch.device('cuda') if feature1.is_cuda else torch.device('cpu'))
        batch_size = labels.shape[0]

        # features = torch.cat([feature1, feature2], dim=0) 
        # labels = torch.cat([labels, labels], dim=0) # size([batch_size * 2 + class_num])

        class_num = prototype.shape[0]
        features = torch.cat([feature1, feature2, prototype], dim=0) # size([batch_size * 2 + class_num, feature_dim])
        class_index = torch.arange(class_num).to(device)
        labels = torch.cat([labels, labels, class_index], dim=0) # size([batch_size * 2 + class_num])

        # features, labels = filter_only_one_sample_class(features, labels, class_num=class_num)

        labels_ = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels_[:2 * batch_size], labels_.T).float().to(device)
        # logits_mask 对角线为0, 其他全1， 为了剔除自己和自己对比
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(2 * batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute logits
        sim = torch.div(
            torch.matmul(features[:2 * batch_size], features.T),
            self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        batch_cls_count = torch.eye(self.class_num)[labels].sum(dim=0).to(device)
        # batch_cls_count = torch.eye(self.class_num)[labels].sum(dim=0)
        # print(batch_cls_count.shape)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        exp_logits /= batch_cls_count[labels].unsqueeze(0) # 使用batch上的标签分布进行balance

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos.mean()
        return loss
