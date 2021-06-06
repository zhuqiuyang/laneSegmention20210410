import torch
import torch.nn as nn
from model.network import ResNet101v2
from model.module import Block

class UNetConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_chans))

        block.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_chans))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, up_mode, padding):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_chans, out_chans, kernel_size=1),
            )
        self.conv_block = UNetConvBlock(in_chans, out_chans, padding, True)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([crop1, up], dim=1)
        out = self.conv_block(out)

        return out

class ResNetUNet(nn.Module):
    def __init__(
        self,
        config
    ):
        super(ResNetUNet, self).__init__()
        self.n_classes = config.NUM_CLASSES
        self.padding = 1
        self.up_mode = 'upconv'
        assert self.up_mode in ('upconv', 'upsample')
        self.encode = ResNet101v2()
        prev_channels = 2048
        self.up_path = nn.ModuleList()
        for i in range(3):
            self.up_path.append(
                UNetUpBlock(prev_channels, prev_channels // 2, self.up_mode, self.padding)
            )
            prev_channels //= 2

        self.cls_conv_block1 = Block(prev_channels, 32)
        self.cls_conv_block2 = Block(32, 16)
        self.last = nn.Conv2d(16, self.n_classes, kernel_size=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.size()[2:]
        blocks = self.encode(x)
        x = blocks[-1]
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 2])
        x = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)(x)
        x = self.cls_conv_block1(x)
        x = self.cls_conv_block2(x)
        x = self.last(x)
        return x
