# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    '''
    ResNet的BasicBlock结构，它有两个卷积层，一个short_cut，具体有没有short_cut是由图片的大小或者通道数来决定的。
    '''
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    '''
    ResNet的BottleNeck结构，与BasicBlock结构相似，同时做了一个改进。
    先做1*1卷积，接着做3*3卷积，最后再做1*1卷积，相当于先缩小通道数，再放大，这个放大倍数一般是4。
    '''
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):
    '''
    作业要求的意思是希望改进这个类后能放到U_Net++中。
    '''
    def __init__(self, in_chans, block, num_block, num_classes=100):
        super().__init__()

        self.block = block
        self.in_channels = 64
        '''
        修改conv1，使输出的feature_map与原图大小完全相同，注意此处要求将kernel_size修改为3
        '''
        self.conv1 = nn.Sequential(
            #nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False),
            #此处原来的代码stride=2，使图片分辨率直接缩小一半，下面maxpooling中stride=2，操作后分辨率又缩小一半，最终分辨率缩小为最初的1/4。
            #这和U-Net++结构中按1/2缩小的效果不一致。
            #改为下面这样：
            nn.Conv2d(in_chans, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # 同时这里要求kenel_size大小设置为3，这是常见的kernel_size改进，因为改进后参数量比较少，且小卷积核效果通常优于大卷积核。
            # 因为希望输出大小与原图完全相同，所以padding应该是same padding，即(3-1)/2 = 1
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2_x(self.pool(f1))
        f3 = self.conv3_x(f2)
        f4 = self.conv4_x(f3)
        f5 = self.conv5_x(f4)
        output = self.avg_pool(f5)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        '''
        提取网络的3个中间层feature_map，要求第一个与原图大小相同，第二个为原图的1/2，第三个为原图的1/4
        '''
        #return output
        return f1,f2,f3
        # 因为第一个卷积conv1的stride=1，所以输出大小是和原图相同的；
        # pooling是降采样2倍，conv2的stride=1，两个操作后输出大小是原图的1/2，所以返回f2大小是原图的1/2；
        # 类似地，f3输出大小是原图1/4。

def resnet18(in_chans):
    return ResNet(in_chans, BasicBlock, [2, 2, 2, 2])

def resnet34(in_chans):
    return ResNet(in_chans, BasicBlock, [3, 4, 6, 3])

def resnet50(in_chans):
    return ResNet(in_chans, BottleNeck, [3, 4, 6, 3])

def resnet101(in_chans):
    return ResNet(in_chans, BottleNeck, [3, 4, 23, 3])

def resnet152(in_chans):
    return ResNet(in_chans, BottleNeck, [3, 8, 36, 3])

"""### ResNet_UNetpp"""

class ConvBlock(nn.Module):

  def __init__(self, in_chans, out_chans, stride):
    super(ConvBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1)
    self.bn1 = nn.BatchNorm2d(out_chans)
    self.relu1 = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(out_chans)
    self.relu2 = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.relu1(self.bn1(self.conv1(x)))
    out = self.relu2(self.bn2(self.conv2(x)))
    return out

'''
将该模块的卷积块修改为BasicBlock
'''
class UpConvBlock(nn.Module):

  def __init__(self, in_chans, bridge_chans_list, out_chans):
    super(UpConvBlock, self).__init__()
    self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
    self.up_4 = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=4, stride=2)
    #将ConvBlock替换为BasicBlock，其中的in_channels，out_channels，stride都是一样的。
    #self.conv_block = ConvBlock(out_chans + sum(bridge_chans_list), out_chans, 1)
    self.conv_block = BasicBlock(out_chans + sum(bridge_chans_list), out_chans, 1)

  def forward(self, x, bridge_list):
    import pdb
    pdb.set_trace()
    x = self.up(x)
    x = torch.cat([x] + bridge_list, dim=1)
    out = self.conv_block(x)
    return out

class ResNet_UNetpp(nn.Module):

  def __init__(self, in_chans=1, n_classes=2, backbone=resnet18):
    super(ResNet_UNetpp, self).__init__()
    
    '''
    利用__init__函数的最后一个参数backbone，来替换下面的三个ConvBlock
    注意feat_chans要进行相应修改(借助expansion)，同时兼容resnet18/34/50/101/152
    feat_chans中可以有相同的数字
    '''
    # feat_chans = [64, 128, 256]
    # self.conv_x00 = ConvBlock(in_chans, feat_chans[0], 1)
    # self.conv_x10 = ConvBlock(feat_chans[0], feat_chans[1], 2)
    # self.conv_x20 = ConvBlock(feat_chans[1], feat_chans[2], 2)
    self.backbone = backbone(in_chans)
    expansion = self.backbone.block.expansion
    #根据ResNet结构参数写出下面的表达
    feat_chans = [64,64*expansion,128*expansion]
    '''
    以下网络结构不允许修改
    '''
    self.conv_x01 = UpConvBlock(feat_chans[1], [feat_chans[0]], feat_chans[0])
    self.conv_x11 = UpConvBlock(feat_chans[2], [feat_chans[1]], feat_chans[1])
    self.conv_x02 = UpConvBlock(feat_chans[1], [feat_chans[0], feat_chans[0]], feat_chans[0])
    
    self.cls_conv_x01 = nn.Conv2d(feat_chans[0], 2, kernel_size=1)
    self.cls_conv_x02 = nn.Conv2d(feat_chans[0], 2, kernel_size=1)

  def forward(self, x):
    '''
    替换为backbone的输出
    '''
    # x00 = self.conv_x00(x)
    # x10 = self.conv_x10(x00)
    # x20 = self.conv_x20(x10)
    x00, x10, x20 = self.backbone(x)
    x01 = self.conv_x01(x10, [x00])
    x11 = self.conv_x11(x20, [x10])
    x02 = self.conv_x02(x11, [x00, x01])
    out01 = self.cls_conv_x01(x01)
    out02 = self.cls_conv_x02(x02)

    '''
    用以下代码打印backbone为resnet34和resnet50时的结果，并截图提交
    '''
    print('x00', x00.shape)
    print('x10', x10.shape)
    print('x20', x20.shape)
    print('x01', x01.shape)
    print('x11', x11.shape)
    print('x02', x02.shape)
    print('out01', out01.shape)
    print('out02', out02.shape)

    return out01, out02

x = torch.randn((2, 1, 224, 224), dtype=torch.float32)
model = ResNet_UNetpp(in_chans=1, backbone=resnet34)
y1, y2 = model(x)
