# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

"""### 建立block"""

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out

"""### 建立layer"""

def make_layers(in_channels, layer_list):
    layers = []
    for out_channels in layer_list:
      layers += [Block(in_channels, out_channels)]
      in_channels = out_channels
    return nn.Sequential(*layers)
            

class Layer(nn.Module):
    def __init__(self, in_channels, layer_list):
        super(Layer, self).__init__()
        self.layer = make_layers(in_channels, layer_list)
    def forward(self, x):
        out = self.layer(x)
        return out

"""### 建立VGG-19BN模型

* 'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
* 'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
* 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
* 'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
"""

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, n_class=21):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
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

        # modify to be compatible with segmentation and classification
        self.fc6 = nn.Linear(512*7*7, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout()

        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout()

        self.score = nn.Linear(4096, n_class)

    def forward(self, x):
        f0 = self.relu1(self.bn1(self.conv1(x)))
        f1 = self.pool1(self.layer1(f0))
        f2 = self.pool2(self.layer2(f1))
        f3 = self.pool3(self.layer3(f2))
        f4 = self.pool4(self.layer4(f3))
        f5 = self.pool5(self.layer5(f4))
        f5 = f5.view(f5.size(0), -1)
        print(f5.shape)
        f6 = self.drop6(self.relu6(self.fc6(f5)))
        f7 = self.drop7(self.relu7(self.fc7(f6)))
        score = self.score(f7)
        return score

vgg_model = VGG()
x = torch.randn((2, 3, 224, 224), dtype=torch.float32)
y = vgg_model(x)
print(y.shape)

"""### 上采样模块的权重初始化"""

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    
    center = kernel_size / 2 - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)

import cv2
x = cv2.imread('cat.png')
x_torch = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
# channels, bias ?
conv_trans = nn.ConvTranspose2d(3, 3, 4, 2, 1)
conv_trans.weight.data = bilinear_kernel(3, 3, 4)
import pdb
pdb.set_trace()
y_torch = conv_trans(x_torch)
y = y_torch.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

x = x.astype(np.uint8)
print(x.shape)
#cv2.imshow('show', x)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

y = y.astype(np.uint8)
print(y.shape)
#cv2.imshow('show', y)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

"""### 建立VGG_19bn_8s模型"""

class VGG_19bn_8s(nn.Module):
    def __init__(self, n_class=21):
        super(VGG_19bn_8s, self).__init__()
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
        
        self.fc6 = nn.Conv2d(512, 4096, 7) # padding=0
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.trans_f4 = nn.Conv2d(512, n_class, 1)
        self.trans_f3 = nn.Conv2d(256, n_class, 1)

        self.up2times = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.up4times = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.up32times = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data = bilinear_kernel(n_class, n_class, m.kernel_size[0])
    
    def forward(self, x):
        f0 = self.relu1(self.bn1(self.conv1(x)))
        f1 = self.pool1(self.layer1(f0))
        f2 = self.pool2(self.layer2(f1))
        f3 = self.pool3(self.layer3(f2))
        f4 = self.pool4(self.layer4(f3))
        f5 = self.pool5(self.layer5(f4))
        
        f6 = self.drop6(self.relu6(self.fc6(f5)))
        f7 = self.score_fr(self.drop7(self.relu7(self.fc7(f6))))
        
        up2_feat = self.up2times(f7)
        h = self.trans_f4(f4)
        h = h[:, :, 5:5 + up2_feat.size(2), 5:5 + up2_feat.size(3)]
        h = h + up2_feat
        
        up4_feat = self.up4times(h)
        h = self.trans_f3(f3)
        h = h[:, :, 9:9 + up4_feat.size(2), 9:9 + up4_feat.size(3)]
        h = h + up4_feat
        
        h = self.up32times(h)
        final_scores = h[:, :, 31:31 + x.size(2), 31:31 + x.size(3)].contiguous()
        
        return final_scores

model = VGG_19bn_8s(21)
x = torch.randn(2, 3, 58, 58)
model.eval()
y_vgg = model(x)
y_vgg.size()

from torchsummary import summary
summary(model, (3, 224, 224))

