import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import tools.joint_transforms as joint_transforms
from models.FGCN import FCN8s
from tools.config import msra10k_path
from tools.datasets import ImageFolder
from tools.misc import (AvgMeter, check_mkdir)

cudnn.benchmark = True

ckpt_path = './ckpt/ckpt_FGCN'
exp_name = 'FGCN'
torch.manual_seed(2019)
torch.cuda.set_device(0)

writer = SummaryWriter('runs/exp_FGCN')

# resnet+fcn
# msra: 8000
#
args = {
    'iter_num': 8000,
    'train_batch_size': 4,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',  # 接着上次的迭代
    'monitor_update': 20
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
    # 可能需要修改吧？
    # transforms.Normalize([0.520, 0.462, 0.431], [0.267, 0.253, 0.247])  # 可能需要修改吧？
])

target_transform = transforms.ToTensor()

train_set = ImageFolder(msra10k_path,
                        True,
                        joint_transform,
                        img_transform,
                        target_transform)
train_loader = DataLoader(train_set,
                          batch_size=args['train_batch_size'],
                          num_workers=12,
                          shuffle=True)

# 使得数值结果更加稳定（numerical stability）。建议使用这个损失函数。
criterion = nn.BCELoss().cuda()
# 将运行记录写下来
log_path = os.path.join(ckpt_path, exp_name,
                        str(datetime.now()) + '.txt')


def main():
    # 显著性用1就可以. 指定的是最后输出的通道数
    net = FCN8s(pretrained=True).cuda()
    
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
    curr_iter = args['last_iter']  # 接着上次训练
    while True:
        loss_record = AvgMeter()
    
        tqdm_loader = tqdm(train_loader)
        for i, data in enumerate(tqdm_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (
                1 - float(curr_iter) / args['iter_num']
            ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (
                1 - float(curr_iter) / args['iter_num']
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
            
            curr_iter += 1

            log = f"[iter {curr_iter}], [total loss {loss_record.avg}], " \
                f"[lr {optimizer.param_groups[1]['lr']}]"
            print(log)

            if i % args['monitor_update'] == 0:
                # ques:
                #   这里应该用loss_record.avg还是iter_loss?
                writer.add_scalar('loss', iter_loss, curr_iter)
            
            open(log_path, 'a').write(log + '\n')

            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(
                    ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(), os.path.join(
                    ckpt_path, exp_name, '%d_optim.pth' % curr_iter))

                print("train over")
                return


if __name__ == '__main__':
    main()
