import torch
import torch.nn.functional as F
from torch import nn

from models.resnext.resnext101 import ResNeXt101


class R2Net(nn.Module):
    def __init__(self):
        super(R2Net, self).__init__()
        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4
        self.ASPP = _ASPP(2048)
        
        self.ARM5_r0 = nn.Sequential(
            nn.Conv2d(2048 + 1, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 1, kernel_size=3, padding=1)
        )
        self.ARM5_s0 = nn.Conv2d(1, 1, kernel_size=1)
        
        self.ARM4_r0 = nn.Sequential(
            nn.Conv2d(1024 + 1 + 1, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 1, kernel_size=3, padding=1)
        )
        self.ARM4_s0 = nn.Conv2d(1, 1, kernel_size=1)
        
        self.ARM3_r0 = nn.Sequential(
            nn.Conv2d(512 + 1 + 1, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 1, kernel_size=3, padding=1)
        )
        self.ARM3_s0 = nn.Conv2d(1, 1, kernel_size=1)
        
        self.ARM2_r0 = nn.Sequential(
            nn.Conv2d(256 + 1 + 1, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 1, kernel_size=3, padding=1)
        )
        self.ARM2_s0 = nn.Conv2d(1, 1, kernel_size=1)
        
        self.ARM1_r0 = nn.Sequential(
            nn.Conv2d(64 + 1 + 1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
        self.ARM1_s0 = nn.Conv2d(1, 1, kernel_size=1)
        
        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True
    
    def forward(self, x):
        # print(x.shape) #300
        layer0 = self.layer0(x)
        # print(layer0.shape) #75
        layer1 = self.layer1(layer0)
        # print(layer1.shape) #75
        layer2 = self.layer2(layer1)
        # print(layer2.shape) #38
        layer3 = self.layer3(layer2)
        # print(layer3.shape) #19
        layer4 = self.layer4(layer3)
        # print(layer4.shape)
        
        output0 = self.ASPP(layer4)
        # print(output0.shape)
        
        ARM5_r0 = self.ARM5_r0(torch.cat((output0, layer4), 1))
        # print(ARM5_r0.shape)
        ARM5_s0 = self.ARM5_s0(ARM5_r0)
        # print(ARM5_s0.shape)
        output5 = ARM5_s0 + output0
        
        # print(output5.shape)
        ARM4_r0 = self.ARM4_r0(torch.cat((
            F.upsample(output5, size=layer3.size()[2:], mode='bilinear'),
            F.upsample(ARM5_r0, size=layer3.size()[2:], mode='bilinear'),
            layer3
        ), 1))
        # print(ARM4_r0.shape)
        ARM4_s0 = self.ARM4_s0(ARM4_r0)
        # print(ARM4_s0.shape)
        output4 = F.upsample(
            output5, size=layer3.size()[2:], mode='bilinear') + ARM4_s0
        # print(output4.shape)
        
        ARM3_r0 = self.ARM3_r0(torch.cat((
            F.upsample(output4, size=layer2.size()[2:], mode='bilinear'),
            F.upsample(ARM4_r0, size=layer2.size()[2:], mode='bilinear'),
            layer2
        ), 1))
        # print(ARM3_r0.shape)
        ARM3_s0 = self.ARM3_s0(ARM3_r0)
        # print(ARM3_s0.shape)
        output3 = F.upsample(
            output4, size=layer2.size()[2:], mode='bilinear') + ARM3_s0
        
        # print(output3.shape)
        ARM2_r0 = self.ARM2_r0(torch.cat((
            F.upsample(output3, size=layer1.size()[2:], mode='bilinear'),
            F.upsample(ARM3_r0, size=layer1.size()[2:], mode='bilinear'),
            layer1
        ), 1))
        # print(ARM2_r0.shape)
        ARM2_s0 = self.ARM2_s0(ARM2_r0)
        # print(ARM2_s0.shape)
        output2 = F.upsample(
            output3, size=layer1.size()[2:], mode='bilinear') + ARM2_s0
        # print(output2.shape)
        
        ARM1_r0 = self.ARM1_r0(torch.cat((
            F.upsample(output2, size=layer0.size()[2:], mode='bilinear'),
            F.upsample(ARM2_r0, size=layer0.size()[2:], mode='bilinear'),
            layer0
        ), 1))
        # print(ARM1_r0.shape)
        ARM1_s0 = self.ARM1_s0(ARM1_r0)
        # print(ARM1_s0.shape)
        output1 = F.upsample(
            output2, size=layer0.size()[2:], mode='bilinear') + ARM1_s0
        output1 = F.upsample(output1, size=x.size()[2:], mode='bilinear')
        # print(output1.shape)
        
        if self.training:
            # nn.BCEWithLogitsLosss使用这个就不需要进行sigmoid
            return output0, output5, output4, output3, output2, output1
        return torch.sigmoid(output1)


# 增大感受野,而且不做pooling损失信息。 每个卷积输出都包含较大范围信息。里面是带孔卷积。
# ASPP是多尺度的，所以对于进景，远景有较好的分割。
class _ASPP(nn.Module):  # 保持不变
    def __init__(self, in_dim):
        super(_ASPP, self).__init__()
        
        down_dim = in_dim // 2
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(down_dim), nn.PReLU()
        )  # 空洞卷积增大感受野，通过dilation然后padding，保持尺寸不变
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4),
            nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6),
            nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim), nn.PReLU()
            # 如果batch=1 ，进行batchnorm会有问题
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim), nn.PReLU(),
            nn.Conv2d(in_dim, 1, kernel_size=1),
            nn.BatchNorm2d(1), nn.PReLU()
        )
    
    def forward(self, x):
        conv1 = self.conv1(x)
        # print(conv1.shape)
        conv2 = self.conv2(x)
        # print(conv2.shape)
        conv3 = self.conv3(x)
        # print(conv3.shape)
        conv4 = self.conv4(x)
        # print(conv4.shape)
        # 对于下面的batch不能设为1
        conv5 = F.upsample(
            self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:],
            mode='bilinear')
        # 如果batch设为1，这里就会有问题。
        # print(conv5.shape)

        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))
