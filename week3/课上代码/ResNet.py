# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    # resnet 18/34
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
    # resnet 50/101/152
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

    def __init__(self, in_chans, block, num_block, num_classes=100):
        super().__init__()

        self.block = block
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # resnet50:[3,4,6,3]
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        import pdb
        #pdb.set_trace()
        # strides=[stride,1,1,1,1,1]
        # [1,1]
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        import pdb
        #pdb.set_trace()
        f1 = self.conv1(x)
        f2 = self.conv2_x(self.pool(f1))
        f3 = self.conv3_x(f2)
        f4 = self.conv4_x(f3)
        f5 = self.conv5_x(f4)
        output = self.avg_pool(f5)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return f1,f2,f3,f4,f5,output

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

x = torch.randn((2, 3, 224, 224), dtype=torch.float32)
model = resnet18(3)
_,_,_,_,_,y = model(x)
print(x.shape)
print(y.shape)

"""### ResNet-UNet"""

class UNetUpBlock(nn.Module):
  def __init__(self, in_chans, bridge_chans, out_chans, up_mode):
    super(UNetUpBlock, self).__init__()
    if up_mode == 'upconv':
      self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
    elif up_mode == 'upsample':
      self.up = nn.Sequential(
        nn.Upsample(mode='bilinear', scale_factor=2),
        nn.Conv2d(in_chans, out_chans, kernel_size=1),
      )
    self.conv_block = BasicBlock(out_chans + bridge_chans, out_chans)

  def center_crop(self, layer, target_size):
    _, _, layer_height, layer_width = layer.size()
    diff_y = (layer_height - target_size[0]) // 2
    diff_x = (layer_width - target_size[1]) // 2
    return layer[:, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])]

  def forward(self, x, bridge):
    up = self.up(x)
    crop = self.center_crop(bridge, up.shape[2:])
    out = torch.cat([crop, up], dim=1)
    out = self.conv_block(out)
    return out

class ResNet_UNet(nn.Module):

  def __init__(self, in_chans=1, n_classes=2, up_mode='upconv'):
    super(ResNet_UNet, self).__init__()
    self.n_classes = n_classes
    self.up_mode = 'upconv'
    assert self.up_mode in ('upconv', 'upsample')

    self.encoder = resnet18(in_chans)
    in_chans = 512 * self.encoder.block.expansion

    self.decoder = nn.ModuleList()
    for i in range(3):
      self.decoder.append(UNetUpBlock(in_chans, in_chans // 2, in_chans // 2, self.up_mode))
      in_chans //= 2
    self.decoder.append(UNetUpBlock(in_chans, 64, 64, self.up_mode))

    self.cls_conv = nn.Conv2d(64, self.n_classes, kernel_size=1)
        
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    # encoding
    f1, f2, f3, f4, f5, _ = self.encoder(x)
    bridges = [f1, f2, f3, f4]
    x = f5

    # decoding
    for i, decode_layer in enumerate(self.decoder):
      x = decode_layer(x, bridges[-i-1])
    
    score = self.cls_conv(x)
    return score
resnet50(1)
x = torch.randn((2, 1, 224, 224), dtype=torch.float32)
unet = ResNet_UNet()
print(x.shape)
y = unet(x)
print(y.shape)
