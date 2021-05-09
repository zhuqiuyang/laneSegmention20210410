# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetConvBlock(nn.Module):
  def __init__(self, in_chans, out_chans, padding, batch_norm):
    super(UNetConvBlock, self).__init__()
    block = []

    block.append(nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=int(padding)))
    if batch_norm:
      block.append(nn.BatchNorm2d(out_chans))
    block.append(nn.ReLU())

    block.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=int(padding)))
    if batch_norm:
      block.append(nn.BatchNorm2d(out_chans))
    block.append(nn.ReLU())

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
    return layer[:, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])]

  def forward(self, x, bridge):
    up = self.up(x)
    crop1 = self.center_crop(bridge, up.shape[2:])
    out = torch.cat([crop1, up], dim=1)
    out = self.conv_block(out)
    return out

class UNet(nn.Module):

  def __init__(self, in_chans=1, n_classes=2, padding=False, up_mode='upconv'):
    super(UNet, self).__init__()
    self.n_classes = n_classes
    self.padding = padding
    self.up_mode = 'upconv'
    assert self.up_mode in ('upconv', 'upsample')

    out_chans = 64
    self.encoder = nn.ModuleList()
    for i in range(5):
      self.encoder.append(UNetConvBlock(in_chans, out_chans, self.padding, batch_norm=False))
      in_chans = out_chans
      out_chans *= 2

    self.decoder = nn.ModuleList()
    for i in range(4):
      self.decoder.append(UNetUpBlock(in_chans, in_chans // 2, self.up_mode, self.padding))
      in_chans //= 2

    self.cls_conv = nn.Conv2d(in_chans, self.n_classes, kernel_size=1)
        
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    # encoding
    bridges = []
    print("encoder")
    for i, encode_layer in enumerate(self.encoder):
      x = encode_layer(x)
      if i < len(self.encoder) - 1:
        bridges.append(x)
        x = F.max_pool2d(x, kernel_size=2)
      print(x.shape)

    # decoding
    print("decoder")
    for i, decode_layer in enumerate(self.decoder):
      x = decode_layer(x, bridges[-i-1])
      print(x.shape)
    
    score = self.cls_conv(x)
    return score

x = torch.randn((2, 1, 572, 572), dtype=torch.float32)
unet = UNet(padding=False)
print(x.shape)
y = unet(x)
print(y.shape)
