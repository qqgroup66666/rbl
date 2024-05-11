

# %%

import math
import os
import numpy as np
from torchvision import datasets
import random

def get_factor_per_cls(cls_num, imb_factor=0.01):
    factors = []
    for cls_idx in range(cls_num):
        factor = (imb_factor**(cls_idx / (cls_num - 1.0)))
        factors.append(factor)
    return factors

def gen_imbalanced_data(data, target, img_num_per_cls):
    lt_data = []
    lt_target = []
    for class_index, the_img_num in enumerate(img_num_per_cls):
        idx = np.where(target == class_index)[0]
        np.random.shuffle(idx)
        selec_idx = idx[:the_img_num]
        lt_data.append(data[selec_idx, ...])
        lt_target.extend([class_index, ] * the_img_num)
    lt_data = np.vstack(lt_data)
    return lt_data, lt_target

def generate_cifar100_lt(im_factor):
    train_dataset = datasets.CIFAR100("./", train=True, download=True)
    test_dataset = datasets.CIFAR100("./", train=False, download=True)

    data = np.concatenate([train_dataset.data, test_dataset.data])
    label = np.array(train_dataset.targets + test_dataset.targets)

    factors = get_factor_per_cls(cls_num=100, imb_factor=im_factor) # 8:1:1, 480:60:60
    print("class distribution:", factors)

    train_data, train_label = [], []
    val_data, val_label, test_data, test_label = [], [], [], [] # 与train同为长尾的
    val_data_1, val_label_1, test_data_1, test_label_1 = [], [], [], [] # 均匀分布的val和test
    for c, f in enumerate(factors):
        idx = np.where(label == c)[0]
        np.random.shuffle(idx)

        nnnn_1 = math.ceil(480 * f)
        train_data.append(data[idx[:nnnn_1], ...])
        train_label.extend([c] * nnnn_1)

        # 长尾的val和test
        nnnn_2 = math.ceil(nnnn_1 + 60 * f)
        nnnn_3 = math.ceil(nnnn_2 + 60 * f)
        val_data.append(data[idx[nnnn_1:nnnn_2], ...])
        test_data.append(data[idx[nnnn_2:nnnn_3], ...])
        val_label.extend([c] * (nnnn_2 - nnnn_1))
        test_label.extend([c] * (nnnn_3 - nnnn_2))

        # 均匀的的val和test
        nnnn_2 = math.ceil(nnnn_1 + 60)
        nnnn_3 = math.ceil(nnnn_2 + 60)
        val_data_1.append(data[idx[nnnn_1:nnnn_2], ...])
        test_data_1.append(data[idx[nnnn_2:nnnn_3], ...])
        val_label_1.extend([c] * (nnnn_2 - nnnn_1))
        test_label_1.extend([c] * (nnnn_3 - nnnn_2))

    train_data = np.concatenate(train_data)
    train_label = np.array(train_label)

    val_data = np.concatenate(val_data)
    val_label = np.array(val_label)
    test_data = np.concatenate(test_data)
    test_label = np.array(test_label)

    val_data_1 = np.concatenate(val_data_1)
    val_label_1 = np.array(val_label_1)
    test_data_1 = np.concatenate(test_data_1)
    test_label_1 = np.array(test_label_1)
    return train_data, train_label, val_data, val_label, test_data, test_label, \
        val_data_1, val_label_1, test_data_1, test_label_1

def generate_cifar10_lt(im_factor):
    train_dataset = datasets.CIFAR10("./", train=True, download=True)
    test_dataset = datasets.CIFAR10("./", train=False, download=True)

    data = np.concatenate([train_dataset.data, test_dataset.data])
    label = np.array(train_dataset.targets + test_dataset.targets)

    factors = get_factor_per_cls(cls_num=10, imb_factor=im_factor) # 8:1:1, 4800:600:600
    print("class distribution:", factors)

    train_data, train_label = [], []
    val_data, val_label, test_data, test_label = [], [], [], [] # 与train同为长尾的
    val_data_1, val_label_1, test_data_1, test_label_1 = [], [], [], [] # 均匀分布的val和test
    for c, f in enumerate(factors):
        idx = np.where(label == c)[0]
        np.random.shuffle(idx)

        nnnn_1 = math.ceil(4800 * f)
        train_data.append(data[idx[:nnnn_1], ...])
        train_label.extend([c] * nnnn_1)

        # 与train同为长尾的
        nnnn_2 = math.ceil(nnnn_1 + 600 * f)
        nnnn_3 = math.ceil(nnnn_2 + 600 * f)
        val_data.append(data[idx[nnnn_1:nnnn_2], ...])
        test_data.append(data[idx[nnnn_2:nnnn_3], ...])
        val_label.extend([c] * (nnnn_2 - nnnn_1))
        test_label.extend([c] * (nnnn_3 - nnnn_2))

        # 均匀的
        nnnn_2 = math.ceil(nnnn_1 + 600)
        nnnn_3 = math.ceil(nnnn_2 + 600)
        val_data_1.append(data[idx[nnnn_1:nnnn_2], ...])
        test_data_1.append(data[idx[nnnn_2:nnnn_3], ...])
        val_label_1.extend([c] * (nnnn_2 - nnnn_1))
        test_label_1.extend([c] * (nnnn_3 - nnnn_2))

    train_data = np.concatenate(train_data)
    train_label = np.array(train_label)

    val_data = np.concatenate(val_data)
    val_label = np.array(val_label)
    test_data = np.concatenate(test_data)
    test_label = np.array(test_label)

    val_data_1 = np.concatenate(val_data_1)
    val_label_1 = np.array(val_label_1)
    test_data_1 = np.concatenate(test_data_1)
    test_label_1 = np.array(test_label_1)
    return train_data, train_label, val_data, val_label, test_data, test_label, \
        val_data_1, val_label_1, test_data_1, test_label_1

# Balanced Softmax setting
def generate_cifar10_Balanced_Softmax_setting(im_factor):
    train_dataset = datasets.CIFAR10("./", train=True, download=True)
    test_dataset = datasets.CIFAR10("./", train=False, download=True)

    data = train_dataset.data
    label = np.array(train_dataset.targets)

    factors = get_factor_per_cls(cls_num=10, imb_factor=im_factor) # 8:1:1, 4800:600:600
    print("class distribution:", factors)

    train_data, train_label = [], []
    for c, f in enumerate(factors):
        idx = np.where(label == c)[0]
        np.random.shuffle(idx)

        nnnn_1 = math.ceil(5000 * f)
        train_data.append(data[idx[:nnnn_1], ...])
        train_label.extend([c] * nnnn_1)

    train_data = np.concatenate(train_data)
    train_label = np.array(train_label)

    return train_data, train_label, test_dataset.data, test_dataset.targets

# Balanced Softmax setting
def generate_cifar100_Balanced_Softmax_setting(im_factor):
    train_dataset = datasets.CIFAR100("./", train=True, download=True)
    test_dataset = datasets.CIFAR100("./", train=False, download=True)

    data = train_dataset.data
    label = np.array(train_dataset.targets)

    factors = get_factor_per_cls(cls_num=100, imb_factor=im_factor) # 8:1:1, 480:60:60
    print("class distribution:", factors)

    train_data, train_label = [], []
    for c, f in enumerate(factors):
        idx = np.where(label == c)[0]
        np.random.shuffle(idx)

        nnnn_1 = math.ceil(500 * f)
        train_data.append(data[idx[:nnnn_1], ...])
        train_label.extend([c] * nnnn_1)

    train_data = np.concatenate(train_data)
    train_label = np.array(train_label)

    return train_data, train_label, test_dataset.data, test_dataset.targets

def main_cifar10(root, IR, with_val):
    if not os.path.exists(root):
        os.mkdir(root)

    if with_val:
        for path in IR:
            if not os.path.exists(os.path.join(root, path)):
                os.mkdir(os.path.join(root, path))
            im = 1 / int(path)

            train_data, train_label, val_data, val_label, test_data, test_label, \
                val_data_1, val_label_1, test_data_1, test_label_1 = generate_cifar10_lt(im_factor=im)

            np.save(os.path.join(root, path, "train.npy"), {
                'data': train_data,
                'targets': train_label
            })
            np.save(os.path.join(root, path, "val.npy"), {
                'data': val_data,
                'targets': val_label
            })
            np.save(os.path.join(root, path, "test.npy"), {
                'data': test_data,
                'targets': test_label
            })

            np.save(os.path.join(root, path, "val_uniform.npy"), {
                'data': val_data_1,
                'targets': val_label_1
            })
            np.save(os.path.join(root, path, "test_uniform.npy"), {
                'data': test_data_1,
                'targets': test_label_1
            })
        return 

    # without val
    root = os.path.join(root, "without_val")
    if not os.path.exists(root):
        os.mkdir(root)

    for path in IR:
        if not os.path.exists(os.path.join(root, path)):
            os.mkdir(os.path.join(root, path))
        im = 1 / int(path)

        train_data, train_label, test_data, test_label = \
            generate_cifar10_Balanced_Softmax_setting(im_factor=im)

        np.save(os.path.join(root, path, "train.npy"), {
            'data': train_data,
            'targets': train_label
        })
        np.save(os.path.join(root, path, "test.npy"), {
            'data': test_data,
            'targets': test_label
        })

def main_cifar100(root, IR, with_val):

    if not os.path.exists(root):
        os.mkdir(root)
    if with_val:
        for path in IR:
            if not os.path.exists(os.path.join(root, path)):
                os.mkdir(os.path.join(root, path))
            im = 1 / int(path)

            train_data, train_label, val_data, val_label, test_data, test_label, \
                val_data_1, val_label_1, test_data_1, test_label_1 = generate_cifar100_lt(im_factor=im)
            np.save(os.path.join(root, path, "train.npy"), {
                'data': train_data,
                'targets': train_label
            })
            np.save(os.path.join(root, path, "val.npy"), {
                'data': val_data,
                'targets': val_label
            })
            np.save(os.path.join(root, path, "test.npy"), {
                'data': test_data,
                'targets': test_label
            })
            np.save(os.path.join(root, path, "val_uniform.npy"), {
                'data': val_data_1,
                'targets': val_label_1
            })
            np.save(os.path.join(root, path, "test_uniform.npy"), {
                'data': test_data_1,
                'targets': test_label_1
            })
        return
    
    # without val
    root = os.path.join(root, "without_val")
    if not os.path.exists(root):
        os.mkdir(root)

    for path in IR:
        if not os.path.exists(os.path.join(root, path)):
            os.mkdir(os.path.join(root, path))
        im = 1 / int(path)

        train_data, train_label, test_data, test_label = \
            generate_cifar100_Balanced_Softmax_setting(im_factor=im)

        np.save(os.path.join(root, path, "train.npy"), {
            'data': train_data,
            'targets': train_label
        })
        np.save(os.path.join(root, path, "test.npy"), {
            'data': test_data,
            'targets': test_label
        })

if __name__ == '__main__':
    main_cifar100(root="cifar-100-lt", IR=["50"], with_val=False)
    # main_cifar10(root="cifar-10-lt", IR=["1"], with_val=False)

