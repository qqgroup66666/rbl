import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random


import torch
import time
import shutil
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
import warnings
import torch.backends.cudnn as cudnn
import random
from randaugment import rand_augment_transform
import torchvision

import argparse
import os


class ImageNetLT(Dataset):

    def __init__(self, root, txt, transform=None, train=True):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.num_classes = 1000
        self.train = train
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.labels)):
            y = self.labels[i]
            self.class_data[y].append(i)

        self.cls_num_list = [len(self.class_data[i]) for i in range(self.num_classes)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.class_balance:
            label = random.randint(0, self.num_classes - 1)
            index = random.choice(self.class_data[label])
            path = self.img_path[index]

        else:
            path = self.img_path[index]
            label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            if self.train:
                sample1 = self.transform[0](sample)
                sample2 = self.transform[1](sample)
                sample3 = self.transform[2](sample)
                return [sample1, sample2, sample3], label  # , index
            else:
                return self.transform(sample), label


if __name__ == "__main__":

    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    txt_train = f'/data1/peifeng/Imagenet2012/ImageNet_LT_train.txt'
    txt_val = f'/data1/peifeng/Imagenet2012/ImageNet_LT_val.txt'
    txt_test = f'/data1/peifeng/Imagenet2012/ImageNet_LT_test.txt'
    data_path = "/data1/peifeng/Imagenet2012"

    val_dataset = ImageNetLT(
        root=data_path,
        txt=txt_val,
        transform=val_transform,
        train=False
    )

    train_dataset = ImageNetLT(
        root=data_path,
        txt=txt_train,
        transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=8, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        num_workers=8, pin_memory=True)

    for i, j in train_loader:
        print(i.shape)
        print(j.shape)
