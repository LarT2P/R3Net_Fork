import torch
import torch.nn.functional as F
import torch.nn as nn

from resnext import ResNeXt101
from modules.DDRNN import RNN


class R3Net(nn.Module):
    def __init__(self):
        super(R3Net, self).__init__()
        
        resnext = ResNeXt101()
        # 对应的五个阶段
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4
        
        # F1, F2, F3
        self.reduce_low = nn.Sequential(
            nn.Conv2d(64 + 256 + 512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )
        
        # F4, F5
        self.reduce_high = nn.Sequential(
            nn.Conv2d(1024 + 2048, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            _ASPP(256)
        )
        
        # 使用高层次特征预测S0
        self.predict0 = nn.Conv2d(256, 1, kernel_size=1)
        
        # Residual1
        self.predict1 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        # Residual2
        self.predict2 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        # Residual3
        self.predict3 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        # Residual4
        self.predict4 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        # Residual5
        self.predict5 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        # Residual6
        self.predict6 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        
        # self.rnn_0 = RNN(1, 3, 1)
        # self.rnn_1 = RNN(1, 3, 1)
        # self.rnn_2 = RNN(1, 3, 1)
        # self.rnn_3 = RNN(1, 3, 1)
        # self.rnn_4 = RNN(1, 3, 1)
        # self.rnn_5 = RNN(1, 3, 1)
        # self.rnn_6 = RNN(1, 3, 1)
        
        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True
    
    def upsample(self, x, size, mode):
        if mode == 'nearest':
            return F.interpolate(x, size=size, mode=mode)
        else:
            return F.interpolate(x, size=size, mode=mode, align_corners=True)
    
    def forward(self, x):
        # 获取五个阶段的特征输出
        layer0 = self.layer0(x)  # x/4 64
        layer1 = self.layer1(layer0)  # x/4 256
        layer2 = self.layer2(layer1)  # x/8 512
        layer3 = self.layer3(layer2)  # x/16 1024
        layer4 = self.layer4(layer3)  # x/32 2048
        
        # print(layer1.size(), layer0.size())
        
        # 前三层的融合特征, 面积放缩到了原图边长的1/4大小
        # layer0_output_w&h
        l0_size = layer0.size()[2:]
        reduce_low = self.reduce_low(torch.cat(
            (layer0,
             self.upsample(layer1, size=l0_size, mode='bilinear'),
             self.upsample(layer2, size=l0_size, mode='bilinear')), 1))
        # 256
        
        # 后两层的融合特征
        l3_size = layer3.size()[2:]
        reduce_high = self.reduce_high(torch.cat(
            (layer3,
             self.upsample(layer4, size=l3_size, mode='bilinear')), 1))
        # 256
        
        # 对融合出来的的高阶特征进一步上采样到第一阶段输出特征图的大小, 也就是低级融合特征的大
        # 小, 准备进一步的拼接
        reduce_high = self.upsample(reduce_high, size=l0_size, mode='bilinear')
        
        # S0 1
        predict0 = self.predict0(reduce_high)
        #  predict0 = self.rnn_0(predict0)
        
        # S1, F=L 1
        predict1 = self.predict1(torch.cat((predict0, reduce_low), 1)) \
                   + predict0
        #  predict1 = self.rnn_1(predict1)
        # S2, F=H 1
        predict2 = self.predict2(torch.cat((predict1, reduce_high), 1)) \
                   + predict1
        #  predict2 = self.rnn_2(predict2)
        # S3, F=L 1
        predict3 = self.predict3(torch.cat((predict2, reduce_low), 1)) \
                   + predict2
        #  predict3 = self.rnn_3(predict3)
        # S4, F=H 1
        predict4 = self.predict4(torch.cat((predict3, reduce_high), 1)) \
                   + predict3
        #  predict4 = self.rnn_4(predict4)
        # S5, F=L 1
        predict5 = self.predict5(torch.cat((predict4, reduce_low), 1)) \
                   + predict4
        #  predict5 = self.rnn_5(predict5)
        # S6, F=H 1
        predict6 = self.predict6(torch.cat((predict5, reduce_high), 1)) \
                   + predict5
        #  predict6 = self.rnn_6(predict6)
        
        # 将所有的预测双线性插值到原图大小
        predict0 = self.upsample(predict0, size=x.size()[2:], mode='bilinear')
        predict1 = self.upsample(predict1, size=x.size()[2:], mode='bilinear')
        predict2 = self.upsample(predict2, size=x.size()[2:], mode='bilinear')
        predict3 = self.upsample(predict3, size=x.size()[2:], mode='bilinear')
        predict4 = self.upsample(predict4, size=x.size()[2:], mode='bilinear')
        predict5 = self.upsample(predict5, size=x.size()[2:], mode='bilinear')
        predict6 = self.upsample(predict6, size=x.size()[2:], mode='bilinear')
        
        # 训练的时候要对7个预测都进行监督(深监督)
        if self.training:
            return predict0, predict1, predict2, predict3, predict4, predict5, predict6
        # 预测的时候只使用最后的预测结果就可以
        return torch.sigmoid(predict6)


class _ASPP(nn.Module):
    #  this module is proposed in deeplabv3 and we use it in all of our baselines
    def __init__(self, in_dim):
        super(_ASPP, self).__init__()
        
        self.upsample = lambda x, size, mode: F.interpolate(
            x, size=size, mode=mode
        )
        
        down_dim = in_dim // 2
        
        # 分流扩张阶段, 为了保证输出与输入大小一致, dilation = padding
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4),
            nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6),
            nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        # 这里虽然和self.conv1代码一致, 但是后续操作不同, 这里实际上感受野是逐渐递增的, 这
        # 里配合后面的平均, 实现了全局的感受野
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        
        # 融合阶段
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim), nn.PReLU()
        )
    
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        # 先对输入整理到NxCx1x1, 再卷积到指定的通道数, 放缩到原输入大小
        conv5 = self.upsample(
            self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:],
            mode='bilinear'
        )
        
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))
