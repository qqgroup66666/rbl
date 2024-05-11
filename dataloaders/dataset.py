from .base_dataset import pil_loader
from .base_dataset import BaseDataset
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks,\
    InstanceHardnessThreshold, NearMiss
from tqdm import tqdm
import torch
import numpy as np
import os
import os.path as osp
import cv2
import pandas as pd
import time
import pickle
from PIL import Image
import lmdb
# *******************
import sys
sys.path.append(os.pardir)
# *******************


def build_lmdb(save_path, metas, commit_interval=1000):
    if not save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with 'lmdb'.")
    if osp.exists(save_path):
        print('`Folder` [{:s}] already exists.'.format(save_path))
        return
    if not osp.exists('/'.join(save_path.split('/')[:-1])):
        print('/'.join(save_path.split('/')[:-1]))
        os.makedirs('/'.join(save_path.split('/')[:-1]))
    data_size_per_img = cv2.imread(metas[0][0], cv2.IMREAD_UNCHANGED).nbytes
    data_size = data_size_per_img * len(metas)
    env = lmdb.open(save_path, map_size=data_size * 10)
    txn = env.begin(write=True)

    shape = dict()

    print('Building lmdb...')
    for i in tqdm(range(len(metas))):
        image_filename = metas[i][0]
        img = pil_loader(filename=image_filename)
        assert img is not None and len(img.shape) == 3

        txn.put(image_filename.encode('ascii'), img.copy(order='C'))
        shape[image_filename] = '{:d}_{:d}_{:d}'.format(
            img.shape[0], img.shape[1], img.shape[2])

        if i % commit_interval == 0:
            txn.commit()
            txn = env.begin(write=True)

    pickle.dump(shape, open(os.path.join(save_path, 'meta_info.pkl'), "wb"))

    txn.commit()
    env.close()
    print('Finish writing lmdb.')


def get_all_files(dir, ext):
    for e in ext:
        if dir.lower().endswith(e):
            return [dir]

    if not osp.isdir(dir):
        return []

    file_list = os.listdir(dir)
    ret = []
    for i in file_list:
        ret += get_all_files(osp.join(dir, i), ext)
    return ret

class Dataset(BaseDataset):
    def __init__(self, args, split='train', aug=True, **kwargs):
        super().__init__(args)
        self.data_dir = osp.join(args.data_dir, split)
        if not 'class2id' in args.keys():
            self.class2id = dict()
            for i in range(args.num_classes):
                self.class2id[str(i)] = i
        else:
            self.class2id = args.get('class2id')

        self.args = args
        self.aug = aug
        aug_type = args.get("aug_type", "CE")
        print("using augmentation:", aug_type)
        
        self.split = split
        if split == 'train':
            self.multi_view_num = args.get("multi_view_num", 1) # 用于监督对比学习
        else:
            self.multi_view_num = 1
            if type(aug_type) == list:
                aug_type = aug_type[0]

        aug_mapping = {
            "val": self.transform_validation(),
            "CE": self.transform_train_ce(),
            "SupCon": self.transform_train_supcon(),
            "SimAug": self.transform_train_SimAug(),
            "cifar_AutoAug_Cutout": self.transform_train_AutoAug_Cutout(),
            "BCL_cls_branch": self.BCL_cls_branch(),
            "transform_imagenet_val": self.transform_imagenet_val(),
            "transform_imagenet_sade_aug": self.transform_imagenet_sade_aug(),
            "transform_imagenet_cls_aug": self.transform_imagenet_cls_aug(),
            "transform_imagenet_sim_aug": self.transform_imagenet_sim_aug(),
            "transform_imagenet_clsstack_aug": self.transform_imagenet_clsstack_aug(),
            "transform_inat_aug": self.transform_inat_aug()
        }
        if type(aug_type) == str: # 多个view全部使用同一种数据增广方式
            self.transform_train_func = [aug_mapping[aug_type]] * self.multi_view_num
        elif type(aug_type) == list:
            assert len(aug_type) == self.multi_view_num
            self.transform_train_func = [aug_mapping[at] for at in aug_type]

        if args.dataset_name == "imagenet-lt-256":
            self.transform_validation_ = self.transform_imagenet_val()
        elif args.dataset_name == "inat2018":
            self.transform_validation_ = self.transform_inat_val()
        else:
            self.transform_validation_ = self.transform_validation()

        self.tmp = None
        self.data = None
        self.targets = None
        self.metas = []

        if self.args.get('npy_style', False): # 若使用npy文件
            self.tmp = np.load(self.data_dir + '.npy', allow_pickle=True).item()
            self.data = self.tmp['data']
            self.targets = self.tmp['targets']
            assert len(self.data) == len(self.targets)

            self.img_list = ['%08d' % i for i in range(len(self.data))]
            for i in range(len(self.targets)):
                cls_id = self.class2id.get(str(self.targets[i]), 0)
                if cls_id < 0:
                    continue
                self.metas.append((self.data[i], np.array(cls_id)))
            args.use_lmdb = False
            self.args.use_lmdb = False
        else: # 使用文件系统
            print(self.data_dir)
            self.img_list = get_all_files(self.data_dir, ['jpg', 'jpeg', 'png'])
            self._gen_metas(self.img_list)

        # 采样方法, 对下标采样
        self._labels = np.array([i[1] for i in self.metas])
        self.resample_index, _ = self.resample(
            np.arange(self._labels.shape[0]).reshape(-1, 1), self._labels
        )
        self.resample_index = self.resample_index.reshape((-1))
        self._num = self.resample_index.shape[0]

        if args.get('use_lmdb', False): # 若使用lmdb
            self.lmdb_dir = osp.join(args.lmdb_dir, split + '.lmdb')
            build_lmdb(self.lmdb_dir, self.metas)
            self.initialized = False
            self._load_image = self._load_image_lmdb
        else:
            self._load_image = self._load_image_pil

        self._cls_num_list = pd.Series(
            self._labels).value_counts().sort_index().values
        self._freq_info = [
            num * 1.0 / sum(self._cls_num_list) for num in self._cls_num_list
        ]
        self._num_classes = len(self._cls_num_list)
        print('class number: ', self._cls_num_list)

    def resample(self, x, y):
        print("Use no sampler.")
        print('%s set has %d images' % (self.split, y.shape[0]))
        return x, y

    def _gen_metas(self, img_list):
        self.metas = []
        n_cls_p = len(set([0] + [i[1]
                      for i in list(self.class2id.items())])) - 1
        for i in img_list:
            cls_id = self.class2id.get(i.split('/')[-2], 0)
            # if cls_id < 0:
            #     continue
            # label_one_hot = np.zeros(n_cls_p)
            # if cls_id > 0:
            #     label_one_hot[cls_id - 1] = 1
            self.metas.append((i, np.array(cls_id)))

    def _init_lmdb(self):
        if not self.initialized:
            env = lmdb.open(self.lmdb_dir, readonly=True,
                            lock=False, readahead=False, meminit=False)
            self.lmdb_txn = env.begin(write=False)
            self.meta_info = pickle.load(
                open(os.path.join(self.lmdb_dir, 'meta_info.pkl'), "rb"))
            self.initialized = True

    def _load_image_lmdb(self, img_filename):
        self._init_lmdb()
        img_buff = self.lmdb_txn.get(img_filename.encode('ascii'))
        C, H, W = [int(i) for i in self.meta_info[img_filename].split('_')]
        img = np.frombuffer(img_buff, dtype=np.uint8).reshape(C, H, W)
        img = Image.fromarray(img)
        return img

    def _load_image_pil(self, img_filename):
        def pil_load(f):
            return Image.open(f).convert('RGB')
        return pil_load(img_filename)

    def get_labels(self):
        return self._labels

    def get_cls_num_list(self):
        return self._cls_num_list

    def get_freq_info(self):
        return self._freq_info

    def get_num_classes(self):
        return self._num_classes

    def __len__(self):
        return self._num

    def __str__(self):
        return self.args.data_dir + '  split=' + str(self.split)

    def _getitem(self, idx, aug_index): 
        # aug_index为None不增广
        # aug_index为 0至multi_view-1, 按照对应self.transform_train_func增广
        image = self._load_image(self.metas[idx][0])
        label = self.metas[idx][1]
        if aug_index == None:
            image = self.transform_validation_(image)
        else:
            image = self.transform_train_func[aug_index](image)
        label = torch.tensor(label)
        return image, label

    def _getitem_npy(self, idx, aug_index):
        # aug_index为None不增广
        # aug_index为 0至multi_view-1, 按照对应self.transform_train_func增广
        image = self.metas[idx][0]
        label = self.metas[idx][1]
        image = Image.fromarray(image)
        if aug_index == None:
            image = self.transform_validation_(image)
        else:
            image = self.transform_train_func[aug_index](image)
        label = torch.tensor(label)
        return image, label

    def __getitem__(self, idx):
        idx = self.resample_index[idx]
        if self.args.get('npy_style', False):
            getitem = self._getitem_npy
        else:
            getitem = self._getitem

        multi_view_data = []
        multi_view_label = []
        for i in range(self.multi_view_num):
            if self.aug:
                image, label = getitem(idx, aug_index=i) # image: [channel, w, h].  label: []
            else:
                image, label = getitem(idx, aug_index=None) # image: [channel, w, h].  label: []
            multi_view_data.append(image.unsqueeze(0))
            multi_view_label.append(label)
        multi_view_data = torch.cat(multi_view_data, dim=0)
        multi_view_label = torch.tensor(multi_view_label).long()
        return multi_view_data, multi_view_label


class TLDataset(Dataset):
    """
    Dataset resampled by TomekLinks
    """

    def resample(self, x, y):
        print("Use TomekLinks sampler.")
        print('Original %s set has %d images' % (self.split, y.shape[0]))
        resampler = TomekLinks(sampling_strategy='majority')
        x, y = resampler.fit_resample(x, y)
        print('Sampled %s set has %d images' % (self.split, y.shape[0]))
        return x, y


class IHTDataset(Dataset):
    """
    Dataset resampled by InstanceHardnessThreshold
    """

    def resample(self, x, y):
        print("Use InstanceHardnessThreshold sampler.")
        print('Original %s set has %d images' % (self.split, y.shape[0]))
        resampler = InstanceHardnessThreshold(sampling_strategy='majority')
        x, y = resampler.fit_resample(x, y)
        print('Sampled %s set has %d images' % (self.split, y.shape[0]))
        return x, y


class NMDataset(Dataset):
    """
    Dataset resampled by NearMiss
    """

    def resample(self, x, y):
        print("Use NearMiss sampler.")
        print('Original %s set has %d images' % (self.split, y.shape[0]))
        resampler = NearMiss(sampling_strategy='majority')
        x, y = resampler.fit_resample(x, y)
        print('Sampled %s set has %d images' % (self.split, y.shape[0]))
        return x, y


class BSDataset(Dataset):
    """
    Dataset resampled by BorderlineSMOTE
    """

    def resample(self, x, y):
        print("Use BorderlineSMOTE sampler.")
        print('Original %s set has %d images' % (self.split, y.shape[0]))
        resampler = BorderlineSMOTE()
        x, y = resampler.fit_resample(x, y)
        print('Sampled %s set has %d images' % (self.split, y.shape[0]))
        return x, y


class ADADataset(Dataset):
    """
    Dataset resampled by ADASYN
    """

    def resample(self, x, y):
        print("Use ADASYN sampler.")
        print('Original %s set has %d images' % (self.split, y.shape[0]))
        resampler = ADASYN(sampling_strategy='minority')
        x, y = resampler.fit_resample(x, y)
        print('Sampled %s set has %d images' % (self.split, y.shape[0]))
        return x, y
