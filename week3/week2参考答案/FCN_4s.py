#coding:utf-8
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchsummary import summary


'''
建立block，输入与输出的尺寸一样，half padding, no strides
stride = 1, padding = (k -1) / 2 = 1
'''
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


'''
建立layer
layer_list :输出通道数是多少
'''
def make_layers(in_channels, layer_list):
    layers = []
    for out_channels in layer_list:
        layers += [Block(in_channels, out_channels)]
        in_channels = out_channels
    return nn.Sequential(*layers)  # 依次顺序执行

class Layer(nn.Module):
    def __init__(self, in_channels, layer_list):
        super(Layer, self).__init__()
        self.layer = make_layers(in_channels, layer_list)

    def forward(self, x):
        out = self.layer(x)
        return out


# 上采样模块的权重初始化，：（(n-d)/n）
def bilinear_kernel(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2  # 取整数 n

    center = kernel_size / 2 - 0.5  # 中心点，（没有0.5 会出现黑边）
    og = np.ogrid[:kernel_size, :kernel_size]  # m*1 和 1*m
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)  # x y 轴
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt  # 赋值
    return torch.from_numpy(weight)  # numpy 转 torch


# 建立VGG_19bn_4s模型
class Vgg19Bn4s(nn.Module):
    def __init__(self, n_class=21):
        super(Vgg19Bn4s, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=100)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = Layer(64, [64])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = Layer(64, [128, 128])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = Layer(128, [256, 256, 256, 256])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = Layer(256, [512, 512, 512, 512])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = Layer(512, [512, 512, 512, 512])
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc6 = nn.Conv2d(512, 4096, 7)  # padding=0 卷积化
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.trans_f4 = nn.Conv2d(512, n_class, 1)  # skip 跳层，通道的转换 pool4
        self.trans_f3 = nn.Conv2d(256, n_class, 1)  # skip 跳层，通道的转换 pool3
        self.trans_f2 = nn.Conv2d(128, n_class, 1)  # skip 跳层，通道的转换 pool2

        # 修改了 self.up4times
        self.up2times_1 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)  # 2倍上采样 , padding=1 控制尺寸的输出
        self.up2times_2 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)  # 2倍上采样 , padding=1 控制尺寸的输出
        self.up2times_2 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)  # 2倍上采样 , padding=1 控制尺寸的输出
        self.up4times = nn.ConvTranspose2d(
            n_class, n_class, 8, stride=4, bias=False)  # 4倍上采样
        self.up8times = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)  # 8倍上采样
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):  # 初始化
                m.weight.data = bilinear_kernel(n_class, n_class, m.kernel_size[0])

    def forward(self, x):
        print('x.size(2)', x.size(2))
        print('x.size(3)', x.size(3))
        f0 = self.relu1(self.bn1(self.conv1(x)))
        f1 = self.pool1(self.layer1(f0))
        f2 = self.pool2(self.layer2(f1))
        f3 = self.pool3(self.layer3(f2))
        f4 = self.pool4(self.layer4(f3))
        f5 = self.pool5(self.layer5(f4))

        f6 = self.drop6(self.relu6(self.fc6(f5)))
        f7 = self.score_fr(self.drop7(self.relu7(self.fc7(f6))))
        print('f7.shape:', f7.shape)  # torch.Size([2, 21, 2, 2])

        up2_feat = self.up2times_1(f7)  # 上采样2倍 torch.Size([2, 21, 6, 6])
        h = self.trans_f4(f4)  # pool4 skip
        print('h.shape:', h.shape)
        print('up2_feat.shape: ', up2_feat.shape)
        h = h[:, :, 5:5 + up2_feat.size(2), 5:5 + up2_feat.size(3)]  # crop
        h = h + up2_feat

        up4_feat = self.up2times_2(h)  # 上采样4倍 = 上采样2倍 + 上采样2倍
        h = self.trans_f3(f3)  # pool3 skip
        print('*' * 30)
        print('h.shape:', h.shape)
        print('up4_feat.shape: ', up4_feat.shape)
        h = h[:, :, 9:9 + up4_feat.size(2), 9:9 + up4_feat.size(3)]  # crop
        h = h + up4_feat

        up8_feat = self.up2times_3(h)  # 上采样8倍 = 上采样2倍 + 上采样2倍 + 上采样2倍
        h = self.trans_f2(f2)  # pool2 skip
        print('*' * 30)
        print('h.shape:', h.shape)
        print('up8_feat.shape: ', up8_feat.shape)
        h = h[:, :, 17:17 + up8_feat.size(2), 17:17 + up8_feat.size(3)]  # crop
        h = h + up8_feat

        h = self.up4times(h)  # 上采样32倍 = 上采样2倍 + 上采样2倍 + 上采样2倍 + 上采样4倍
        print('*' * 30)
        print('h.shape:', h.shape)
        # final_scores = h[:, :, 31:31 + x.size(2), 31:31 + x.size(3)].contiguous()  # crop
        final_scores = h[:, :, 33:33 + x.size(2), 33:33 + x.size(3)].contiguous()  # crop

        return final_scores


if __name__ == '__main__':
    # 验证 FCN-4s
    model = Vgg19Bn4s(21)
    x = torch.randn(2, 3, 58, 58) # 58 + 198 = 256 是32的倍数 ，不是32的倍数，会有偏差
    model.eval()
    y_vgg = model(x)
    print('out shape: ', y_vgg.size())

    # 查看模型结构
    # summary(model, (3, 224, 224))
