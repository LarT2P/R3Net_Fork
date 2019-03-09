import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import tools.joint_transforms as joint_transforms
from models.FGCN import FCN8s, FGCN8s
from tools.config import msra10k_path
from tools.datasets import ImageFolder
from tools.misc import (AvgMeter, check_mkdir)

# cudnn.benchmark = True

ckpt_path = './ckpt/ckpt_FGCN'
exp_name = 'FGCN'
torch.manual_seed(2019)
torch.cuda.set_device(0)

writer = SummaryWriter('runs/exp_FGCN')

# resnet+fcn
# msra: 8000
#
args = {
    'train_batch_size': 1,
    # 'iter_num': 0,
    # 'last_iter': 0,
    'epoch_num': 5,
    'last_epoch': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',  # 接着上次的迭代
    'monitor_update': 20,
}

# joint_transforms下面的Compose,RandomCrop,RandomHorizontallyFlip,RandomRotate函数
# 下面表示joint_trasnform.Compose会使用下面的RandomCrop,RandomHorizontallyFlip(),RandomRotate三个函数
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomCrop(640, 640),  # crop的size
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])
# 为了方便进行数据的操作，pytorch团队提供了一个torchvision.transforms包，我们可以用transforms进行以下操作：
# PIL.Image/numpy.ndarray与Tensor的相互转化；
# 归一化；
# 对PIL.Image进行裁剪、缩放等操作。
# 通常，在使用torchvision.transforms，我们通常使用transforms.Compose将transforms组合在一起。
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

target_transform = transforms.ToTensor()

train_set = ImageFolder(msra10k_path,
                        True,
                        joint_transform,
                        img_transform,
                        target_transform)
train_loader = DataLoader(train_set,
                          batch_size=args['train_batch_size'],
                          num_workers=4,
                          shuffle=True)
# 将运行记录写下来
log_name = str(datetime.now()) + '.txt'
log_path = os.path.join(ckpt_path, exp_name, log_name)


def main():
    # 显著性用1就可以. 指定的是最后输出的通道数
    net = FGCN8s(pretrained=True).cuda()

    # 对SGD进行配置,注意有两组params参数。 一个是对bias另一个是对其他的weights
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters()
                    if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters()
                    if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(
            os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(
            os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        # bias的学习率大于weights的
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)
    print("main over")


def train(net, optimizer):
    # curr_iter = args['last_iter']  # 接着上次训练
    curr_epoch = args['last_epoch']
    
    # 使得数值结果更加稳定（numerical stability）。建议使用这个损失函数。
    criterion = nn.BCELoss().cuda()
    
    # 总的迭代周期
    for iter_epoch in range(curr_epoch, args['epoch_num']):
        print(f"现在是周期{iter_epoch}")
        loss_record = AvgMeter()

        tqdm_loader = tqdm(train_loader)
        for i, data in enumerate(tqdm_loader):
            # 每周期衰减一次
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (
                1 - float(iter_epoch) / args['epoch_num']
            ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (
                1 - float(iter_epoch) / args['epoch_num']
            ) ** args['lr_decay']

            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            iter_loss = loss.item()
            loss_record.update(iter_loss, batch_size)
    
            log = f"[epoch {curr_epoch} iter {iter}], [total loss {loss_record.avg}], " \
                f"[lr {optimizer.param_groups[1]['lr']}]"
            print(log)

            if i % args['monitor_update'] == 0:
                # ques:
                #   这里应该用loss_record.avg还是iter_loss?
                curr_iter = curr_epoch * len(train_loader) + i
                writer.add_scalar('loss', iter_loss, curr_iter)
    
            with open(log_path, 'a') as log_file:
                log_file.write(log + '\n')
    
    # 所有的迭代结束
    torch.save(net.state_dict(), os.path.join(
        ckpt_path, exp_name, '%d.pth' % args['epoch_num']))
    torch.save(optimizer.state_dict(), os.path.join(
        ckpt_path, exp_name, '%d_optim.pth' % args['epoch_num']))
    
    print("train over")


if __name__ == '__main__':
    main()
