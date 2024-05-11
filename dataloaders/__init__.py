import torch
import os
from torch.utils.data import DataLoader
from .dataset import (Dataset, TLDataset, IHTDataset, 
        NMDataset, BSDataset, ADADataset)
from .sampler import StratifiedSampler, MultiClassStratifiedSampler


def get_datasets(args, train_aug=True):

    resampler_type = args.get('resampler_type', "None")
    if resampler_type == "None":
        TrainSet = Dataset
    elif resampler_type == 'TL':
        TrainSet = TLDataset
    elif resampler_type == 'IHT':
        TrainSet = IHTDataset
    elif resampler_type == 'NM':
        TrainSet = NMDataset
    elif resampler_type == 'BS':
        TrainSet = BSDataset
    elif resampler_type == 'ADA':
        TrainSet = ADADataset
    else:
        raise RuntimeError('Unknown sampler_type: %s'%resampler_type)

    if os.path.exists(os.path.join(args.data_dir, "test_uniform.npy")):
        train_set = TrainSet(args, split='train', aug=train_aug)
        val_set = Dataset(args, split='val', aug=False)
        test_set = Dataset(args, split='test', aug=False)
        val_uniform_set = Dataset(args, split='val_uniform', aug=False)
        test_uniform_set = Dataset(args, split='test_uniform', aug=False)
    else:
        train_set = TrainSet(args, split='train', aug=train_aug)
        try:
            test_set = Dataset(args, split='test', aug=False)
        except:
            test_set = None
        try:
            val_set = Dataset(args, split='val', aug=False)
        except:
            val_set = None
        val_uniform_set = None
        test_uniform_set = None

        # if args.dataset_name == "imagenet-lt-256":
        #     val_set = test_set
        #     print("-"*100)
        #     print("用test暂代val, 加速看结果")
        #     print("-"*100)
    return train_set, val_set, test_set, val_uniform_set, test_uniform_set

def multi_view_collate(items):
    images = torch.cat([i[0] for i in items], dim=0)
    labels = torch.cat([i[1] for i in items], dim=0)
    return images, labels

def get_data_loaders(train_set,
                     val_set,
                     test_set,
                     train_batch_size,
                     test_batch_size,
                     num_workers=16,
                     distributed=False):
    
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None
    train_loader = DataLoader(train_set,
                            batch_size=train_batch_size,
                            shuffle=(train_sampler is None),
                            sampler=train_sampler,
                            num_workers=num_workers,
                            collate_fn=multi_view_collate,
                            drop_last=True,
                            pin_memory=True)
    val_loader = DataLoader(val_set,
                            batch_size=test_batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=multi_view_collate,
                            pin_memory=True)
    test_loader = DataLoader(test_set,
                             batch_size=test_batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                            collate_fn=multi_view_collate,
                            pin_memory=True)
    return train_loader, val_loader, test_loader

__all__ = ['Dataset', 'get_datasets', 'get_data_loaders', 'StratifiedSampler']
