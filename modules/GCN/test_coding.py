"""
用来测试代码用的文件
"""
import argparse
import time

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from skimage.segmentation import slic

from lib.grid_graph import adjacency, distance_sklearn_metrics
from pygcn.models import GCN
from pygcn.utils import accuracy, encode_onehot

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(f"参数列表:{args}")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# load image
img = np.array(Image.open(
    "./data/images/ILSVRC2012_test_00000196.jpg").convert('RGB'))
gt = np.array(Image.open(
    "./data/masks/ILSVRC2012_test_00000196.png").convert('L'))

# 需要把输入的图片数据先进行超像素的划分.
segments = slic(
    img,
    n_segments=200,
    compactness=10,
    max_iter=10,
    sigma=1,
    spacing=None,
    multichannel=True,
    convert2lab=None,
    enforce_connectivity=False,
    min_size_factor=0.5,
    max_size_factor=3,
    slic_zero=False
)
print("SLIC over")
# todo:
#  因为超像素分割出来的超像素节点并不是定数, 所以可以考虑添加伪节点来补到一致
#  这个可以在同时处理多个图像的时候考虑
# note:
#   每次超像素分割对于同一幅图像, 结果也是不同的, 如何避免这种情况下造成的影响

## 构造节点X: 将每个超像素作为一个节点
# 总的超像素数量, amax方法会返回最大的类别值
nodes_num = sp.amax(segments) + 1
nodes_labels = np.zeros((nodes_num, 1))
nodes_features = np.zeros((nodes_num, 3))
nodes_center = np.zeros((nodes_num, 2))
for i in range(nodes_num):
    ## 获取节点真值
    nodes_labels[i] = np.mean(gt[segments == i], axis=0) / 255
    ## 计算节点特征: 每个超像素内部的像素特征均值作为超像素的特征NxFi features
    nodes_features[i] = np.mean(img[segments == i, :], axis=0)
    ## 获取超像素的中心位置
    pixels_in = np.sum(segments == i)  # 超像素包含像素数量
    nodes_center[i] = np.sum(np.argwhere(segments == i), axis=0) / pixels_in
print("node over")

## 获取图结构的邻接矩阵
dist, idx = distance_sklearn_metrics(nodes_center, k=8)
dist = (dist - dist.min()) / (dist.max() - dist.min())
adj = adjacency(dist, idx).todense()

adj = torch.tensor(adj)
features = torch.tensor(nodes_features)
labels = np.zeros_like(nodes_labels)
labels[nodes_labels > 0] = 1


def encode_onehot(labels):
    return (np.unique(labels) == labels[:]).astype(np.integer)


# labels = encode_onehot(labels)
labels = torch.tensor(labels)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=2,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    # note:
    #   可以考虑之后将所有的图像转化为图节点文件存起来, 之后只需要索引即可.
    labels = labels.cuda().long()


def train(epoch):
    global labels
    
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    labels = labels.view(1, -1).squeeze()
    loss_train = F.nll_loss(output, labels)
    acc_train = accuracy(output, labels)
    loss_train.backward()
    optimizer.step()
    
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          # 'loss_val: {:.4f}'.format(loss_val.item()),
          # 'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
# test()
