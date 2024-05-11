import numpy as np
from torch.utils.data import Dataset
import os
from torchvision import transforms
from dataloaders import custom_transforms as tr
from abc import ABC, abstractmethod
import cv2
from PIL import Image, ImageFile
import scipy.io as scio
from .randaugment import rand_augment_transform
from PIL import ImageFilter
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True
def pil_loader(filename, label=False):
    ext = (os.path.splitext(filename)[-1]).lower()
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        img = Image.open(filename)
        if not label:
            img = img.convert('RGB')
            img = np.array(img).astype(dtype=np.uint8)
            img = img[:,:,::-1]  #convert to BGR
        else:
            if img.mode != 'L' and img.mode != 'P':
                img = img.convert('L')
            img = np.array(img).astype(dtype=np.uint8)
    elif ext == '.mat':
        img = scio.loadmat(filename)
    elif ext == '.npy':
        img = np.load(filename, allow_pickle=True)
    else:
        raise NotImplementedError('Unsupported file type %s'%ext)

    return img

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class BaseDataset(Dataset, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ignore_index = 255
        self.input_size = self.args.input_size
        self.mean = np.array(self.args.norm_params.mean) # [0-255]
        self.std = np.array(self.args.norm_params.std) # [0-255]
        self.normalize = transforms.Normalize(self.mean / 255, self.std / 255)

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __str__(self):
        pass
    
    @staticmethod
    def modify_commandline_options(parser,istrain=False):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    def transform_train_ce(self):
        temp = []
        temp.append(transforms.Resize(size=self.input_size))
        temp.append(transforms.RandomHorizontalFlip())
        # temp.append(transforms.RandomRotation(15))
        temp.append(transforms.RandomCrop(self.input_size))
        temp.append(transforms.ToTensor())
        temp.append(self.normalize)
        composed_transforms = transforms.Compose(temp)
        return composed_transforms

    def transform_train_supcon(self):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=self.input_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            self.normalize,
        ])
        return train_transform

    def transform_train_SimAug(self):
        # s is the strength of color distortion.

        kernal_size = [int(0.1 * i) if int(0.1 * i) % 2 == 1 else int(0.1 * i) + 1 for i in self.input_size]
        s = 0.5
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=self.input_size, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernal_size)], p=0.5),
            transforms.ToTensor(),
            self.normalize
        ])

        return train_transform

    def transform_train_AutoAug_Cutout(self):
        transform = transforms.Compose([
            transforms.RandomCrop(self.input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            tr.CIFAR10Policy(), # AutoAug
            transforms.ToTensor(),
            tr.Cutout(n_holes=1, length=16),
            self.normalize,
        ])
        return transform

    def transform_validation(self):
        temp = []
        temp.append(transforms.Resize(size=self.input_size))
        temp.append(transforms.ToTensor())
        temp.append(self.normalize)
        composed_transforms = transforms.Compose(temp)
        return composed_transforms

    def BCL_cls_branch(self):
        randaug_m = 10
        randaug_n = 2
        ra_params = dict(
            translate_const=int(self.input_size[0] * 0.45), 
            img_mean=tuple([
                min(255, round(255 * x)) for x in self.mean
            ]), 
        )
        temp = [
            transforms.RandomResizedCrop(size=self.input_size, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            rand_augment_transform(
                'rand-n{}-m{}-mstd0.5'.format(randaug_n, randaug_m), ra_params),
            transforms.ToTensor(),
            self.normalize,
        ]
        return transforms.Compose(temp)

    def transform_imagenet_sade_aug(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def transform_imagenet_sim_aug(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def transform_imagenet_cls_aug(self):
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=int(224 * 0.45), 
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean])
        )
        randaug_n = 2
        randaug_m = 10
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(randaug_n, randaug_m), ra_params),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def transform_imagenet_clsstack_aug(self):
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=int(224 * 0.45), 
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean])
        )
        randaug_n = 2
        randaug_m = 10
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(randaug_n, randaug_m), ra_params),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def transform_imagenet_val(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def transform_inat_aug(self):
        return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
            ])

    def transform_inat_val(self):
        return  transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
        ])


