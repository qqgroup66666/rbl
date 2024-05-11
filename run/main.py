# %%
import time
import random
import torch
import numpy as np
import json
import pytest
from easydict import EasyDict as edict


@pytest.mark.skip()
def load_json(config_path):
    with open(config_path, 'r') as f:
        args = json.load(f)
        args = edict(args)
    return args

import sys
import shutil
import os
sys.path.append(os.pardir)

from copy import deepcopy
from torchvision import datasets
from models import generate_net, LinearClassifier
import torch.nn.functional as F
from dataloaders import get_data_loaders
from dataloaders import get_datasets
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
from utils import load_criterion, lr_decay, forward, get_log_file_path, neural_collpase_metric
from tqdm import tqdm
from collections import Counter
import torch.multiprocessing as mp
import torch.distributed as dist

def test_correction(train_loader, val_loader, test_loader, class_num, model, correct):

    acc, ce_loss, precision, recall, f1, \
        val_preds_torch, val_labels_torch = test(model, train_loader, class_num, des_str="train", train_cls_num_list=train_loader.dataset._cls_num_list)

    acc, ce_loss, precision, recall, f1, \
        val_preds_torch, val_labels_torch = test(model, val_loader, class_num, des_str="val", train_cls_num_list=train_loader.dataset._cls_num_list)

    print("-" * 100)

    acc, ce_loss, precision, recall, f1, \
        test_preds_torch, test_labels_torch = test(model, test_loader, class_num, des_str="test", train_cls_num_list=train_loader.dataset._cls_num_list)

    print()
    print("-"*100)
    if not correct:
        return 

def get_metric(preds_torch, labels_torch, class_num, train_cls_num_list=None):

    if train_cls_num_list is not None:
        train_cls_num_list = torch.tensor(train_cls_num_list)
        many_shot = train_cls_num_list > 100
        medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list >= 20)
        few_shot = train_cls_num_list < 20

        one_hot = torch.stack([labels_torch == i for i in range(class_num)], dim=0)
        pred_res = torch.argmax(preds_torch, dim=1)
        every_class_acc = [torch.mean((pred_res[one_hot[c, :]] == c).float()) for c in range(class_num)]
        every_class_acc = torch.tensor(every_class_acc)

        many_shot_acc = torch.mean(every_class_acc[many_shot])
        medium_shot_acc = torch.mean(every_class_acc[medium_shot])
        few_shot_acc = torch.mean(every_class_acc[few_shot])
        many_shot_acc = many_shot_acc.item()
        medium_shot_acc = medium_shot_acc.item()
        few_shot_acc = few_shot_acc.item()
    
    ce_loss = (-torch.log(preds_torch[torch.arange(preds_torch.shape[0]), labels_torch])).mean()

    preds = preds_torch.detach().cpu().numpy()
    labels = labels_torch.detach().cpu().numpy()

    y_pred = np.argmax(preds, axis=1)
    acc = accuracy_score(y_true=labels, y_pred=y_pred, normalize=True)
    precision = precision_score(y_true=labels, y_pred=y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true=labels, y_pred=y_pred, average="macro")
    f1 = f1_score(y_true=labels, y_pred=y_pred, average="macro")

    if train_cls_num_list is not None:
        return acc, ce_loss, precision, recall, f1, many_shot_acc, medium_shot_acc, few_shot_acc
    else:
        return acc, ce_loss, precision, recall, f1

def print_metric(des_str, auc_mu_=None, auc_ski=None, auc_mine=None, acc=None, precision=None, \
    recall=None, f1=None, predict_entropy=None, many_shot_acc=None, medium_shot_acc=None, few_shot_acc=None):
    metrics = [auc_mu_, auc_ski, auc_mine, acc, precision, recall, f1, predict_entropy, many_shot_acc, medium_shot_acc, few_shot_acc]
    metric_names = ["auc_mu_", "auc_ski", "auc_mine", "acc", "precision", "recall", "f1", "predict_entropy", "many_shot_acc", "medium_shot_acc", "few_shot_acc"]
    for metric_name, metric in zip(metric_names, metrics):
        if metric is not None and metric_name == "acc":
            print(des_str, metric_name + ": %.4f" % (metric))

def test(model, test_loader, class_num, des_str, require_feature=False, train_cls_num_list=None):

    val_loss_sum = 0
    labels = []
    preds = []
    features = []
    softmax = nn.Softmax(dim=1)
    model.eval()
    test_loader = tqdm(test_loader)
    with torch.no_grad():
        for img, lbl in test_loader:
            
            if img.shape[0] <= 1:
                continue 
            img = img.cuda()
            lbl = lbl.cuda()
            out = model(img)
            if require_feature:
                features.append(model.forward_feature(img))
            out = softmax(out)
            preds.append(out)
            labels.append(lbl.squeeze(-1))

    if require_feature:
        features = torch.cat(features, dim=0)

    labels_torch = torch.cat(labels, dim=0)
    preds_torch = torch.cat(preds, dim=0)

    if train_cls_num_list is not None:
        acc, ce_loss, precision, recall, f1, many_shot_acc, medium_shot_acc, few_shot_acc = get_metric(preds_torch, labels_torch, class_num, train_cls_num_list)
        print(des_str, "many shot ACC: %.4f" % (many_shot_acc))
        print(des_str, "medium shot ACC: %.4f" % (medium_shot_acc))
        if not math.isnan(few_shot_acc):
            print(des_str, "few shot ACC: %.4f" % (few_shot_acc))
    else:
        acc, ce_loss, precision, recall, f1 = get_metric(preds_torch, labels_torch, class_num, train_cls_num_list)

    print(des_str, "ACC: %.4f" % (acc))
    return acc, ce_loss, precision, recall, f1, preds_torch, labels_torch

def train_classifier(args, encoder, classifier, train_loader, val_loader, test_loader, for_valid):
    max_acc = 0
    class_num = args.model.num_classes
    ce = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD([
        {
            'params': classifier.parameters(),
            'lr': 5e-1,
            'momentum': 0.9,
            'weight_decay': 0
        }
    ])

    epoch_num = 100
    if for_valid:
        epoch_num = 1

    for epoch in range(epoch_num):
        encoder.train()
        classifier.train()
        train_loss_sum = 0
        sample_num = 0

        train_bar = tqdm(train_loader)
        for img, lbl in train_bar:
            if img.shape[0] < args.training.train_batch_size:
                continue

            img = img.cuda()
            lbl = lbl.cuda()
            with torch.no_grad():
                out = encoder(img)
            out = classifier(out)
            loss = ce(out, lbl)
            des = {"classifier epoch": epoch+1, "loss": loss.item()}

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_sum += loss.item() * img.shape[0]
            sample_num += img.shape[0]
            des_str = ""
            for k, v in des.items():
                if type(v) == int:
                    des_str += "%s: %d " % (k, v)
                elif type(v) == float:
                    des_str += "%s: %.4f " % (k, v)
                else:
                    raise RuntimeError("??")
            train_bar.set_description(des_str)

        train_mean_loss = train_loss_sum / sample_num
        print("classifier mean loss: %.4f" % (train_mean_loss))

        model = nn.Sequential(
            encoder,
            classifier
        )
        auc_mu_, auc_ski, auc_mine, acc, precision, recall, f1, predict_entropy, \
            preds_torch, labels_torch = test(model, val_loader, class_num, des_str="val", \
                train_cls_num_list=train_loader.dataset._cls_num_list)

        if not for_valid:
            # 验证集比较最大指标
            if max_acc < acc:
                max_acc = acc
                best_model = deepcopy(classifier.state_dict())
                best_model_epoch = epoch
                print("max val acc")
        print("-" * 100)
    
    if not for_valid:
        model.load_state_dict(best_model)
        test_correction(train_loader, val_loader, test_loader, class_num, model, correct=False)
    return max_acc

def train(args, SEED, log_root=None, gpu=None):

    distributed = args.get("distributed", False)

    if distributed:
        args.rank = gpu
        dist.init_process_group(backend="nccl", init_method=args.dist_url,
                                    world_size=args.gpu_num, rank=args.rank)

    stdout = sys.stdout
    stderr = sys.stderr
    if log_root is not None:
        # 重定向
        begin = time.time()
        from time import sleep
        sleep(random.random())
        log_file_path = get_log_file_path(log_root)
        print("log_file_path:", log_file_path)
        log_file = open(log_file_path, "w+")
        sys.stdout = log_file
        sys.stderr = log_file

        print("log file path:", log_file_path)
        # 最优模型存储位置
        best_model_path = os.path.join(log_file_path + "best_model.pth")

        print("best model path:", best_model_path)

    print("-"*100)
    print(args)
    print("-"*100)
    print("Random SEED:", SEED)
    print("-"*100)

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    method = args.training.loss_type
    assert method in [
        "CE", "FocalLoss", "CB", "LDAM", "BalancedSoftmax",
        "MAUC_square", "MAUC_exp", "MAUC_hinge", 
        "AUC_mu_square", "AUC_mu_exp", "AUC_mu_hinge", "BCE",
        "SCL", "MSL", "BCL", "TCL", "ResLT", "SADE", "softf1",
        "PL", "MarginCalibration", "LogitAdjustment2"
    ]
    print("-"*100)
    print("method:", method)
    print("-"*100)
    print("Dataset:", args.dataset.dataset_name)
    print("-"*100)

    test_batch_size = args.training.test_batch_size
    train_batch_size = args.training.train_batch_size
    print("train batch size:", train_batch_size)

    # val_set, test_set是长尾的，val_uniform_set, test_uniform_set是均匀的
    train_set, val_set, test_set, val_uniform_set, test_uniform_set = get_datasets(args.dataset)
    class_num = len(train_set._cls_num_list)

    print("-"*100)
    print("Traning Set Imbalance Ratio is %.4f." % (max(train_set._cls_num_list) / min(train_set._cls_num_list)))
    print("-"*100)
    num_worker = args.dataset.get("num_worker", 32)

    if distributed:
        train_batch_size = int(train_batch_size / args.gpu_num)
        num_worker = int((num_worker + args.gpu_num - 1) / args.gpu_num)

    if val_set is not None:
        print("There are train, val, test, val_uniform, test_uniform, 5 dataset provided.")
        train_loader, val_loader, test_loader = get_data_loaders(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_worker,
            distributed=distributed
        )

        train_loader, val_uniform_loader, test_uniform_loader = get_data_loaders(
            train_set=train_set,
            val_set=val_uniform_set,
            test_set=test_uniform_set,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_worker,
            distributed=distributed
        )

        use_unifrom_test_and_val = args.dataset.get("use_unifrom_test_and_val", False)
        if use_unifrom_test_and_val:
            print("Use unifrom val set and test set.")
            val_loader = val_uniform_loader
            test_loader = test_uniform_loader
        else:
            print("Use val set and test set that have same distributin as tran set.")
    else:
        print("There are train, test, 2 dataset provided.")
        print("No Valid Dataset! Use Test Set as Valid Set!")
        train_loader, val_loader, test_loader = get_data_loaders(
            train_set=train_set,
            val_set=test_set,
            test_set=test_set,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_worker
        )
    if test_set is None:
        print("test_set is None")
        test_loader = val_loader

    criterion, criterion_CE = load_criterion(method, args, train_set._cls_num_list)

    model_use = args.model.get("model_use", "classifier")
    assert model_use in ("feature_exactor", "BalancedSupConModel", "classifier", \
        "TargetSupConModel", "ResLTModel", "SADEModel", "PLBackbone", "PLBackbone_", \
        "MarginCalibrationBackbone", "neural_collpase", "PLBackbone_factor", "PLPostHocModel", \
        "FixedPermutation_Rotation", "Ablation_CE", "Ablation_LD", "Ablation_Fixed", "Ablation_RBL"
    )

    metric_name = args.training.metric_name
    assert metric_name in ("AUCmu", "ACC")
    print("The evaluation metric is", metric_name)

    model = generate_net(args.model, train_set._cls_num_list)

    if distributed:
        torch.cuda.set_device(args.rank)
        model.cuda(args.rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank], find_unused_parameters=True)
    else:
        model = model.cuda()

    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=1)

    save_all_models = args.training.get("save_all_models", False)
    if save_all_models:
        save_path_every_epoch_model = log_file_path + "_model"
        os.mkdir(save_path_every_epoch_model)
        print("Save models in every epoch:", save_all_models)

    opt = args.training.get("opt", "Adam")
    assert opt in ("SGD", "Adam")
    print("Use optimizer:", opt)

    maxmize_loss = args.training.get("maxmize_loss", False)
    print("Gradient Desent:", not maxmize_loss)
    if opt == "SGD":
        optimizer = torch.optim.SGD([
            {
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
                # 'params': model.backbone.parameters(),
                'params': model.parameters(),
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
                'lr': args.training.lr,
                'momentum': args.training.momentum,
                'weight_decay': args.training.weight_decay,
                'nesterov': args.training.nesterov,
                'maximize': maxmize_loss
            }
        ])
    else:
        optimizer = torch.optim.Adam([
            {
                'params': model.parameters(),
                'lr': args.training.lr,
                'momentum': args.training.momentum,
                'weight_decay': args.training.weight_decay,
                'maximize': maxmize_loss
            }
        ])

    max_eval_metric = 0
    best_model = None
    best_model_epoch = None

    early_stop_epoch = args.training.get("early_stop_epoch", 10000)
    early_stop_epoch = 1
    print("Training model. Training_epoch_num: {}. Early stop epoch: {}.".format(args.training.epoch_num, early_stop_epoch))

    for epoch in range(args.training.epoch_num):
        if epoch > early_stop_epoch:
            print("early stop epoch:", epoch)
            break

        if args.training.get("pretrained_encoder_path", None) != None: # load预训练特征提取器
            pretrain_encoder_path = args.training.get("pretrained_encoder_path")
            print("using pretrained feature extractor:", pretrain_encoder_path)
            state_dict = torch.load(pretrain_encoder_path)
            model.load_state_dict(state_dict)
            break

        model.train()
        train_loss_sum = 0
        sample_num = 0
        train_features = []
        preds = []
        labels = []

        if distributed:
            train_loader.sampler.set_epoch(epoch)

        lr_decay(args, method, optimizer, model, epoch)
        train_bar = tqdm(train_loader)
        for img, lbl in train_bar:
            out, loss, des = forward(args, method, epoch, criterion, model, img, lbl)

            preds.append(out)
            labels.append(lbl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * img.shape[0]
            sample_num += img.shape[0]
            des_str = ""
            for k, v in des.items():
                if type(v) == int:
                    des_str += "%s: %d " % (k, v)
                elif type(v) == float:
                    des_str += "%s: %.4f " % (k, v)
                else:
                    raise RuntimeError("??")
            train_bar.set_description(des_str)

        with torch.no_grad():
            labels = torch.cat(labels, dim=0).cuda()
            preds = torch.cat(preds, dim=0).cuda()

            y_pred = np.argmax(preds.cpu().detach().numpy(), axis=1)
            train_acc = accuracy_score(y_true=labels.cpu().detach().numpy(), y_pred=y_pred, normalize=True)

            preds = softmax(preds)
            train_mean_loss = train_loss_sum / sample_num

        print("train_mean_loss: %.4f" % (train_mean_loss))
        print("train_acc: %.4f" % (train_acc))
        acc, ce_loss, precision, recall, f1, \
            preds_torch, labels_torch = test(
                model, val_loader, class_num, \
                train_cls_num_list=train_set._cls_num_list, des_str="val"
            )

        if metric_name == "AUCmu":
            eval_metric = auc_mu_
        if metric_name == "ACC":
            eval_metric = acc
        # 验证集比较最大指标
        if max_eval_metric < eval_metric:
            max_eval_metric = eval_metric
            best_model = deepcopy(model.state_dict())
            best_model_epoch = epoch
            print("max val", metric_name)
        print("-" * 100)

        if save_all_models:
            torch.save(model.state_dict(), os.path.join(save_path_every_epoch_model, str(epoch) + ".pth"))

    # test....
    print("max val metric model in epoch{}.".format(best_model_epoch))
    model.load_state_dict(best_model)
    test_correction(train_loader, val_loader, test_loader, class_num, model, correct=False)
    if log_root is not None:
            torch.save(best_model, best_model_path)
    print("method: ", method)

    if log_root is not None:
        end = time.time()
        print("training time:", (end - begin))
        log_file.close()
        sys.stdout = stdout
        sys.stderr = stderr

def train_Contrastive(log_root):
    config_root = "configs/"

    configs_names = [
        ("cifar100_pl_posthoc.json", "pl_posthoc"),
        ("cifar10_pl_posthoc.json", "pl_posthoc"),
    ]

    dataset_IR_path = {
        "cifar10":[
            ("50", "./datasets/cifar-lt/cifar-10-lt/without_val/50", "cifar10-50_noval"),
            ("100", "./datasets/cifar-lt/cifar-10-lt/without_val/100", "cifar10-100_noval"),
            ("200", "./datasets/cifar-lt/cifar-10-lt/without_val/200", "cifar10-200_noval"),
        ],
        "cifar100":[
            ("50", "./datasets/cifar-lt/cifar-100-lt/without_val/50", "cifar100-50_noval"),
            ("100", "./datasets/cifar-lt/cifar-100-lt/without_val/100", "cifar100-100_noval"),
            ("200", "./datasets/cifar-lt/cifar-100-lt/without_val/200", "cifar100-200_noval"),
        ]
    }

    for configs_name, method_name in configs_names:
        dataset_name__ = configs_name.split("_")[0]
        for IR, dataset_path, dataset_name in dataset_IR_path[dataset_name__]:
            config_path = os.path.join(config_root, configs_name)
            args = load_json(config_path)
            args.dataset.data_dir = dataset_path
            SEED = 378288
            log_path = os.path.join(log_root, method_name, dataset_name)
            if not os.path.exists(log_path):
                os.makedirs(log_path)

            if "imagenet-lt" in dataset_name__:
                args.dataset.lmdb_dir = os.path.join(args.dataset.data_dir, "lmdb")

            print("config_path:", config_path)
            print("log_root:", log_path)
            print()

            # train(args=args, SEED=SEED, log_root=log_path)
            train(args=args, SEED=SEED)

if __name__ == '__main__':
    log_root = "./logs/"
    train_Contrastive(log_root)
