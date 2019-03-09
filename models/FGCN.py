import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.GCN.lib.grid_graph import adjacency, distance_sklearn_metrics
from modules.GCN.pygcn.models import GCN
from modules.ResNet import resnet152
from modules.VGG import VGGNet


class FCN32s(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = VGGNet()
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2,
                                          padding=1, dilation=1,
                                          output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                                          padding=1, dilation=1,
                                          output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                                          padding=1, dilation=1,
                                          output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                                          padding=1, dilation=1,
                                          output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                                          padding=1, dilation=1,
                                          output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']

        score = self.bn1(self.relu(self.deconv1(x5)))
        score = self.bn2(self.relu(self.deconv2(score)))
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        return score


class FCN16s(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = VGGNet()
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2,
                                          padding=1, dilation=1,
                                          output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                                          padding=1, dilation=1,
                                          output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                                          padding=1, dilation=1,
                                          output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                                          padding=1, dilation=1,
                                          output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                                          padding=1, dilation=1,
                                          output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']
        x4 = output['x4']

        score = self.relu(self.deconv1(x5))
        score = self.bn1(score + x4)
        score = self.bn2(self.relu(self.deconv2(score)))
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        return score


class FCN8s(nn.Module):

    def __init__(self, n_class=1, pretrained=True):
        super().__init__()

        self.n_class = n_class
        net_modules = list(resnet152(pretrained=pretrained).children())
        self.layer_div2 = nn.Sequential(*net_modules[:3])
        self.layer_div4 = nn.Sequential(*net_modules[3:5])
        self.layer_div8 = nn.Sequential(net_modules[5])
        self.layer_div16 = nn.Sequential(net_modules[6])
        self.layer_div32 = nn.Sequential(net_modules[7])

        self.upsample = F.interpolate

        self.relu = nn.ReLU(inplace=True)
        # self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2,
        #                                   padding=1, dilation=1,
        #                                   output_padding=1)

        self.upconv1 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)

        # self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2,
        #                                   padding=1, dilation=1,
        #                                   output_padding=1)
        self.upconv2 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(512)

        # self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
        #                                   padding=1, dilation=1,
        #                                   output_padding=1)
        self.upconv3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # self.deconv4 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2,
        #                                   padding=1, dilation=1,
        #                                   output_padding=1)
        self.upconv4 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        # self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
        #                                   padding=1, dilation=1,
        #                                   output_padding=1)
        self.upconv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        x_2 = self.layer_div2(x)
        x_4 = self.layer_div4(x_2)
        x_8 = self.layer_div8(x_4)
        x_16 = self.layer_div16(x_8)
        x_32 = self.layer_div32(x_16)

        # score = self.relu(self.deconv1(x_32))
        score = self.relu(self.upconv1(self.upsample(
            x_32, scale_factor=2, mode='bilinear')))
        score = self.bn1(score + x_16)

        # score = self.relu(self.deconv2(score))
        score = self.relu(self.upconv2(self.upsample(
            score, scale_factor=2, mode='bilinear')))
        score = self.bn2(score + x_8)

        # score = self.bn3(self.relu(self.deconv3(score)))
        score = self.relu(self.upconv3(self.upsample(
            score, scale_factor=2, mode='bilinear')))
        score = self.bn3(score)

        # score = self.bn4(self.relu(self.deconv4(score)))
        score = self.relu(self.upconv4(self.upsample(
            score, scale_factor=2, mode='bilinear')))
        score = self.bn4(score)

        # score = self.bn5(self.relu(self.deconv5(score)))
        score = self.relu(self.upconv5(self.upsample(
            score, scale_factor=2, mode='bilinear')))
        score = self.bn5(score)

        score = self.classifier(score)

        return torch.sigmoid(score)


class FGCN8s(nn.Module):
    
    def __init__(self, n_class=1, pretrained=True):
        super().__init__()
        
        self.n_class = n_class
        net_modules = list(resnet152(pretrained=pretrained).children())
        self.layer_div2 = nn.Sequential(*net_modules[:3])
        self.layer_div4 = nn.Sequential(*net_modules[3:5])
        self.layer_div8 = nn.Sequential(net_modules[5])
        self.layer_div16 = nn.Sequential(net_modules[6])
        self.layer_div32 = nn.Sequential(net_modules[7])
        self.reduce_conv32 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 128, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 32, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.reduce_conv16 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 32, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.reduce_conv8 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 32, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        
        self.upsample = F.interpolate
        
        self.relu = nn.ReLU(inplace=True)
        
        self.upconv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.upconv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.upconv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.upconv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        
        self.upconv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
        
        self.gcnlayers = GCN(nfeat=32,
                             nhid=32,
                             nclass=32,
                             dropout=0.5)
        
        self.adj = self.define_adj().cuda()
    
    def forward(self, x):
        x_2 = self.layer_div2(x)
        x_4 = self.layer_div4(x_2)
        x_8 = self.layer_div8(x_4)
        x_16 = self.layer_div16(x_8)
        x_32 = self.layer_div32(x_16)
        x_32 = self.reduce_conv32(x_32)
        
        N, C, H, W = x_32.size()
        # 必须设定batch=1
        x_32 = x_32.squeeze(0)
        # w, h, c
        #  x_32 = x_32.transpose(0, 2)
        # wxh, c
        #  x_32 = x_32.view(-1, C)
        # 我怀疑这里不能主要是因为不能固定C处, 可以考虑变形后再转置
        x_32 = x_32.view(C, -1)
        x_32 = x_32.transpose(0, 1)
        
        x_32 = self.gcnlayers(x_32, self.adj)
        # wxh, c
        x_32 = x_32.view(W, H, C)
        # c, h, w
        x_32 = x_32.transpose(0, 2)
        # n, c, h, w
        x_32 = x_32.unsqueeze(0)
        
        score = self.relu(self.upconv1(self.upsample(
            x_32, scale_factor=2, mode='bilinear')))
        x_16 = self.reduce_conv16(x_16)
        score = self.bn1(score + x_16)
        
        score = self.relu(self.upconv2(self.upsample(
            score, scale_factor=2, mode='bilinear')))
        x_8 = self.reduce_conv8(x_8)
        score = self.bn2(score + x_8)
        
        score = self.relu(self.upconv3(self.upsample(
            score, scale_factor=2, mode='bilinear')))
        score = self.bn3(score)
        
        score = self.relu(self.upconv4(self.upsample(
            score, scale_factor=2, mode='bilinear')))
        score = self.bn4(score)
        
        score = self.relu(self.upconv5(self.upsample(
            score, scale_factor=2, mode='bilinear')))
        score = self.bn5(score)
        
        score = self.classifier(score)
        
        return torch.sigmoid(score)
    
    def define_adj(self):
        nodes_num = 20 * 20
        nodes_center = np.array([[x, y] for x in range(20) for y in range(20)])
        
        ## 获取图结构的邻接矩阵
        dist, idx = distance_sklearn_metrics(nodes_center, k=8)
        dist = (dist - dist.min()) / (dist.max() - dist.min())
        adj = adjacency(dist, idx).todense()
        adj = torch.tensor(adj)
        
        return adj


class FCNs(nn.Module):
    
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = VGGNet()
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2,
                                          padding=1, dilation=1,
                                          output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                                          padding=1, dilation=1,
                                          output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                                          padding=1, dilation=1,
                                          output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                                          padding=1, dilation=1,
                                          output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                                          padding=1, dilation=1,
                                          output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
        # classifier is 1x1 conv, to reduce channels from 32 to n_class
    
    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']
        x4 = output['x4']
        x3 = output['x3']
        x2 = output['x2']
        x1 = output['x1']

        score = self.bn1(self.relu(self.deconv1(x5)))
        score = score + x4
        score = self.bn2(self.relu(self.deconv2(score)))
        score = score + x3
        score = self.bn3(self.relu(self.deconv3(score)))
        score = score + x2
        score = self.bn4(self.relu(self.deconv4(score)))
        score = score + x1
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        return score


if __name__ == "__main__":
    pass
