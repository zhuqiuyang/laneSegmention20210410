import torch
from torch._C import Size
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    '''
    空洞空间金字塔池化(Atrous Spatial Pyramid Pooling)在给定的输入上以不同采样率的空洞卷积
    并行采样，相当于以多个比例捕捉图像的上下文。
    '''
    def __init__(self, in_chans, out_chans, rate=1):
        super(ASPP, self).__init__()
        # 以不同的采样率预制空洞卷积（通过调整dilation实现）
        # 1x1卷积——无空洞
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )
        # 3x3卷积——空洞6
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )
        # 3x3卷积——空洞12
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )
        # 3x3卷积——空洞18
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )
        # 全局平均池化——获取图像层级特征
        self.branch5_avg = nn.AdaptiveAvgPool2d(1)
        # 1x1的conv、bn、relu——用于处理平均池化所得的特征图
        self.branch5_conv = nn.Conv2d(in_chans, out_chans, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(out_chans)
        self.branch5_relu = nn.ReLU(inplace=True)
        # 1x1的conv、bn、relu——用于处理concat所得的特征图
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_chans*5, out_chans, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 获取size——用于上采样的时候确定上采样到多大
        b, c, h, w = x.size()
        # 一个1x1的卷积
        conv1x1 = self.branch1(x)
        # 三个3x3的空洞卷积
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # 一个平均池化
        global_feature = self.branch5_avg(x)
        # 对平均池化所得的特征图进行处理
        global_feature = self.branch5_relu(self.branch5_bn(self.branch5_conv(global_feature)))
        # 将平均池化+卷积处理后的特征图上采样到原始x的输入大小
        global_feature = F.interpolate(global_feature, (h, w), None, 'bilinear', True)
        # 把所有特征图cat在一起（包括1x1、三组3x3、平均池化+1x1），cat通道的维度
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # 最后再连一个1x1卷积，把cat翻了5倍之后的通道数缩减回来
        result = self.conv_cat(feature_cat)
        return result

