import torch
from torch.utils.data import Sampler, DataLoader
import numpy as np
import _pickle as pk
from collections import Counter
from sklearn.utils import shuffle
import math
import pandas as pd

import pdb 
from copy import copy
import random

class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, 
                class_vector, 
                batch_size, 
                rpos=1, 
                rneg=4, 
                random_state=7):
        self.class_vector = class_vector
        self.batch_size = batch_size

        self.rpos = rpos
        self.rneg = rneg

        # y = [类别标签1, ....]
        if isinstance(class_vector, torch.Tensor):
            y = class_vector.cpu().numpy()
        else:
            y = np.array(class_vector)

        # {0:类0样本数, ...}
        self.y_counter = Counter(y)
        self.data = pd.DataFrame({'y': y})
        # only implemented for binary classification, 1:pos, 0:neg
        if len(self.y_counter.keys()) == 2:

            ratio = (rneg, rpos)

            # 按照ratio计算每个batch中每类样本数目
            self.class_batch_size = {
                k: math.ceil(batch_size * ratio[k] / sum(ratio))
                for k in self.y_counter.keys()
            }

            # 
            if rpos / rneg > self.y_counter[1] / self.y_counter[0]:
                # 需要加多少正例
                add_pos = math.ceil(rpos / rneg * self.y_counter[0]) - self.y_counter[1]

                print("-" * 50)
                print("To balance ratio, add %d pos imgs (with replace = True)" % add_pos)
                print("-" * 50)

                pos_samples = self.data[self.data.y == 1].sample(add_pos, replace=True)

                assert pos_samples.shape[0] == add_pos

                self.data = self.data.append(pos_samples, ignore_index=False)

            else:
                add_neg = math.ceil(rneg / rpos * self.y_counter[1]) - self.y_counter[0]

                print("-" * 50)
                print("To balance ratio, add %d neg imgs repeatly" % add_neg)
                print("-" * 50)

                neg_samples = self.data[self.data.y == 0].sample(add_neg, replace=True)

                assert neg_samples.shape[0] == add_neg

                self.data = self.data.append(neg_samples, ignore_index=False)
        else:
            raise RuntimeError(">???")
        print("-" * 50)
        print("after complementary the ratio, having %d images" % self.data.shape[0])
        print("-" * 50)

        self.real_batch_size = int(sum(self.class_batch_size.values()))

    def gen_sample_array(self):
        
        def sample_class(group):
            n = self.class_batch_size[group.name]
            return group.sample(n) if group.shape[0] >= n else group.sample(n, replace=True)

        # sampling for each batch
        data = self.data.copy()
        data['idx'] = data.index
        data = data.reset_index()

        result = []
        while True:
            try:
                batch = data.groupby('y', group_keys=False).apply(sample_class)
                assert len(batch) == self.real_batch_size, 'not enough instances!'
            except (ValueError, AssertionError, AttributeError) as e:
                break
            result.extend(shuffle(batch.idx))
            data.drop(index=batch.index, inplace=True)
        return result

    def __iter__(self):
        self.index_list = self.gen_sample_array()
        return iter(self.index_list)

    def __len__(self):
        try:
            l = len(self.index_list)
        except:
            l = len(self.class_vector)
        return l

class Bucket:
    def __init__(self, container, name, need_shuffle=True):
        assert type(container) == list
        self.container = container
        self.pointer = 0
        self.name = name
        self.need_shuffle = need_shuffle # 每次取完一次是否需要重新打乱
        if self.need_shuffle:
            self.reshuffle()

    def getSome(self, num):
        if self.pointer + num >= len(self.container):
            num -= (len(self.container) - self.pointer)
            res = copy(self.container[self.pointer:])
            if self.need_shuffle:
                self.reshuffle()
            res = res + copy(self.container[:num])
            self.pointer = num
        else:
            res = copy(self.container[self.pointer:self.pointer+num])
            self.pointer += num
        return res

    def reshuffle(self):
        self.pointer = 0
        random.shuffle(self.container)
    
    def __str__(self):
        return self.name

class MultiClassStratifiedSampler(Sampler):
    def __init__(self, 
                class_vector,
                batch_size,
                batch_class_num,
                fixed_class_sample_index=False):
        self.class_vector = class_vector
        self.batch_size = batch_size
        self.batch_class_num = batch_class_num
        self.each_class_batch_num = math.ceil(batch_size/batch_class_num) # 类别均衡，每个batch内每个类都有一样数目的样本
        self.real_batch_size = self.each_class_batch_num * batch_class_num

        if isinstance(class_vector, torch.Tensor):
            self.y = class_vector.cpu().numpy()
        else:
            self.y = np.array(class_vector)

        # {0:类0样本数, ...}
        self.y_counter = Counter(self.y)
        self.class_num = len(self.y_counter.keys())

         # 固定每个batch中，类别
        self.fixed_class_sample_index = fixed_class_sample_index
        if self.fixed_class_sample_index and batch_class_num == self.class_num:
            self.fixed_class_sample_index = fixed_class_sample_index
        elif self.fixed_class_sample_index and batch_class_num != self.class_num:
            raise RuntimeError("?")

        self.samples_num = self.y.shape[0]

        index = np.arange(self.y.shape[0])
        class_index = [list(index[self.y==i]) for i in range(self.class_num)]
        each_class_index_bucket = [Bucket(class_index_, "class{}_mask".format(i)) for i, class_index_ in enumerate(class_index)]
        if self.fixed_class_sample_index:
            self.class_bucket = Bucket(each_class_index_bucket, "class", need_shuffle=False)
        else:
            self.class_bucket = Bucket(each_class_index_bucket, "class", need_shuffle=True)
        self.index_list = self.ge_index_list()

    def getIndex(self):
        if not self.fixed_class_sample_index:
            raise RuntimeError("?")

        class_index = [[(i, j) for j in range(self.class_num) if j != i] for i in range(self.class_num)]
        ccc = []
        for i in class_index:
            ccc += i
        class_index = torch.tensor(ccc).cuda()

        labels = torch.arange(self.class_num).unsqueeze(1). \
            expand(torch.Size((self.class_num, self.each_class_batch_num))).reshape(-1).cuda()

        sample_index = torch.arange(self.real_batch_size).cuda()
        each_class_sample_index = [ sample_index[labels==i] for i in range(self.class_num)]

        N = torch.tensor([i.shape[0] for i in each_class_sample_index]).cuda()

        def cartesian_product(a, b):
            la = a.shape[0]
            lb = b.shape[0]
            num = la*lb
            remained_dim = a.shape[1:]
            a = a.unsqueeze(1).expand(torch.Size((la, lb, *remained_dim))).reshape(-1, *remained_dim)
            b = b.unsqueeze(0).expand(torch.Size((la, lb, *remained_dim))).reshape(-1, *remained_dim)
            return torch.stack([a, b], dim=1)

        class_ij_index = torch.cat([
            ij.unsqueeze(0).expand(torch.Size((n_i*n_j, 2))) for (n_i, n_j), ij in zip(N[class_index], class_index)
        ], dim=0)

        sample_ab_index = torch.cat([
            cartesian_product(each_class_sample_index[i], each_class_sample_index[j])
            for (i,j) in class_index
        ], dim=0)

        return torch.cat([class_ij_index, sample_ab_index], dim=1).cuda(), N

    def ge_index_list(self):
        index_list = []
        while len(index_list) < self.y.shape[0]:
            batch = []
            for b in self.class_bucket.getSome(self.batch_class_num):
                batch += b.getSome(self.each_class_batch_num)
            if not self.fixed_class_sample_index:
                random.shuffle(batch)
            index_list += batch
        return index_list

    def __iter__(self):
        self.index_list = self.ge_index_list()
        return iter(self.index_list)

    def __len__(self):
        try:
            return len(self.index_list)
        except AttributeError:
            return self.y.shape[0]