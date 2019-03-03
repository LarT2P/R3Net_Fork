# 如果用服务器就用更大的patch输入到网络中进行计算

import datetime
import os

import torch
import torch.nn.functional as functional
from torch import nn, optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import tools.joint_transforms as joint_transforms
from models.model_R2 import R2Net
from tools.config import msra10k_path
from tools.datasets import ImageFolder
from tools.misc import (AvgMeter, check_mkdir)

# 注意采用的损失函数

cudnn.benchmark = True

torch.manual_seed(2019)
torch.cuda.set_device(0)

ckpt_path = './ckpt/ckpt_R2'
exp_name = 'R2Net'

args = {
    'iter_num': 6000,
    'train_batch_size': 2,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',  # 接着上次的迭代
    'dataset': msra10k_path
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

train_set = ImageFolder(args['dataset'],
                        joint_transform,
                        img_transform,
                        target_transform)
train_loader = DataLoader(train_set,
                          batch_size=args['train_batch_size'],
                          num_workers=12,
                          shuffle=True)
# nn.BCELoss 需要手动加上一个 Sigmoid 层，这里是结合了两者，这样做能够利用 log_sum_exp，
# 使得数值结果更加稳定（numerical stability）。建议使用这个损失函数。
criterion = nn.BCEWithLogitsLoss().cuda()
# 将运行记录写下来
log_path = os.path.join(ckpt_path, exp_name,
                        str(datetime.datetime.now()) + '.txt')


def main():
    net = R2Net().cuda().train()
    # eval()时，pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。
    
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


def train(net, optimizer):
    curr_iter = args['last_iter']  # 接着上次训练
    while True:
        total_loss_record, loss0_record, loss1_record, loss2_record \
            = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss3_record, loss4_record, loss5_record \
            = AvgMeter(), AvgMeter(), AvgMeter()
        
        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (
                1 - float(curr_iter) / args['iter_num']
            ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (
                1 - float(curr_iter) / args['iter_num']
            ) ** args['lr_decay']
            
            inputs, labels, _ = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            
            # multi尺度
            labels1 = functional.interpolate(labels, size=160, mode='bilinear')
            # resnet添加的这一句。如果是vgg就删掉
            labels2 = functional.interpolate(labels, size=80, mode='bilinear')
            labels3 = functional.interpolate(labels, size=40, mode='bilinear')
            labels4 = functional.interpolate(labels, size=20, mode='bilinear')
            labels5 = functional.interpolate(labels, size=20, mode='bilinear')
            
            optimizer.zero_grad()
            outputs0, outputs1, outputs2, outputs3, outputs4, outputs5 = \
                net(inputs)
            
            # loss0最大，loss6最小
            loss0 = criterion(outputs0, labels5)
            loss1 = criterion(outputs1, labels4)
            loss2 = criterion(outputs2, labels3)
            loss3 = criterion(outputs3, labels2)
            loss4 = criterion(outputs4, labels1)
            loss5 = criterion(outputs5, labels)
            
            total_loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5
            total_loss.backward()  # 对6个都进行反向传播
            optimizer.step()
            
            total_loss_record.update(total_loss.data[0], batch_size)
            loss0_record.update(loss0.data[0], batch_size)
            loss1_record.update(loss1.data[0], batch_size)
            loss2_record.update(loss2.data[0], batch_size)
            loss3_record.update(loss3.data[0], batch_size)
            loss4_record.update(loss4.data[0], batch_size)
            loss5_record.update(loss5.data[0], batch_size)
            
            curr_iter += 1
            
            log = '[iter %d], [total loss %.5f], [loss0 %.5f], [loss1 %.5f], ' \
                  '[loss2 %.5f], [loss3 %.5f], [loss4 %.5f], [loss5 %.5f], ' \
                  '[lr %.13f]' % \
                  (curr_iter, total_loss_record.avg, loss0_record.avg,
                   loss1_record.avg, loss2_record.avg,
                   loss3_record.avg, loss4_record.avg, loss5_record.avg,
                   optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')
            
            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(
                    ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(
                               ckpt_path, exp_name, '%d_optim.pth' % curr_iter))
                return


if __name__ == '__main__':
    main()
