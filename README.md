## 1. RBL

Inspired by the ETF structure of Neural Collapse phenomenon, a framework for balanced representation learning based on orthogonal matrix optimization is proposed, addressing the performance bottleneck caused by classifiers unable to learn balanced representations, particularly in imbalanced multiclass scenarios. The introduced learnable orthogonal matrix enables the learning of balanced features in any direction. Experimental results demonstrate the significant improvement in classifier generalization achieved by this approach.

> Gao Peifeng, Qianqian Xu, Peisong Wen, Zhiyong Yang, Huiyang Shao, Qingming Huang. Feature Directions Matter: Long-Tailed Learning via Rotated Balanced Representation. ICML 2023

## 2. Environment Dependencies and Installation

This project requires several Python packages, which can be installed directly using the following command:

```python 
pip3 install -r requirements.txt
pip install git+https://github.com/Lezcano/geotorch/
```

## 3. Running Examples

### Datasets

Two datasets are involved, namely:

+ CIFAR10
+ CIFAR100

These datasets are balanced, and this project provides long-tail versions of the corresponding datasets generated based on specific requirements. They are located in the `./run/data` directory.

### Training

Training parameters are documented in the configuration files located in the `./run/configs` directory. Each configuration file is named according to the specified method and training set
The `./run/configs/` directory includes 2 configuration files, corresponding to the training configurations for the two datasets:

+ ./run/configs/cifar100_pl_posthoc.json
+ ./run/configs/cifar10_pl_posthoc.json

To execute, type:

```python
CUDA_VISIBLE_DEVICES=0
cd ./run
python main.py
```

which shall sequentially train the following setting: 

+ CIFAR100 with Imbalance Ratio 50
+ CIFAR100 with Imbalance Ratio 100
+ CIFAR100 with Imbalance Ratio 200
+ CIFAR10 with Imbalance Ratio 50
+ CIFAR10 with Imbalance Ratio 100
+ CIFAR10 with Imbalance Ratio 200

```bash
epoch: 1 loss: 4.0809 : 100%|██████████████████████████████████████████████████| 49/49 [00:05<00:00,  8.24it/s]
train_mean_loss: 4.2198
train_acc: 0.0482
100%|██████████████████████████████████████████████| 20/20 [00:02<00:00,  8.29it/s]
val many shot ACC: 0.0366
val medium shot ACC: 0.0367
val few shot ACC: 0.0224
val ACC: 0.0342
max val ACC
```

### Logs

Here are some setting reproduction logs:

+ ./cifar100-200.log
+ ./cifar100-100.log
+ ./cifar100-50.log
+ ./cifar10-200.log
+ ./cifar10-100.log
+ ./cifar10-50.log
+ ./imagenet-lt.log