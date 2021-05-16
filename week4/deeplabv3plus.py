import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

bn_mom = 0.0003

# 预先训练模型地址
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
}

# same空洞卷积
# 对于k=3的卷积，通过设定padding=1*atrous,保证添加空洞后的3x3卷积，输入输出feature map同样大小
def conv3x3(in_planes, out_planes, stride=1, atrous=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1 * atrous, dilation=atrous, bias=False)

# 通过 same 空洞卷积实现BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_chans, out_chans, stride=1, atrous=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 使用自定义的same 空洞卷积
        self.conv1 = conv3x3(in_chans, out_chans, stride, atrous)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(out_chans, out_chans)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# 实现带有空洞卷积的Bottleneck
# 这个bottleneck结构，
# 在resnet 50的block1中串连使用了3个，block2中串连使用了4个，block3中串连使用了6个，block4中串连使用了3个。
# 在resnet 101的block1中串连使用了3个，block2中串连使用了4个，block3中串连使用了24个，block4中串连使用了3个。
# 在resnet 152的block1中串连使用了3个，block2中串连使用了8个，block3中串连使用了36个，block4中串连使用了3个。
# 所以，当我们定block1,block2,block3,block4分别为[3,4,6,3]时，就对应resnet50
# 所以，当我们定block1,block2,block3,block4分别为[3,4,24,3]时，就对应resnet101
# 所以，当我们定block1,block2,block3,block4分别为[3,8,36,3]时，就对应resnet152

class Bottleneck(nn.Module):
    # bottleneck block中，有三个卷积层,分别是：C1:1x1conv,C2:3x3conv,C3:1x1conv
    # C1的输入featue map 的channel=4C,输处feature map 的channel=C
    # C2的输入featue map 的channel=C,输处feature map 的channel=C
    # C3的输入featue map 的channel=C,输处feature map 的channel=4C
    # expansion:定义瓶颈处的feature map，C2的输入输出feature map 的 channel是非瓶颈处的channel的1/4
    expansion = 4

    def __init__(self, in_chans, out_chans, stride=1, atrous=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 这里in_chans是out_chans的4倍，在make_layer函数里有实现，大概在本代码164行左右
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chans)
        # same空洞卷积
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=stride,
                               padding=1 * atrous, dilation=atrous, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chans * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# 定义完整的空洞残差网络
class ResNet_Atrous(nn.Module):
    # 当layers=[3,4,6,3]时，block为bottlenet时，就生成resnet50
    def __init__(self, block, layers, atrous=None, os=16):
        super(ResNet_Atrous, self).__init__()
        self.block = block
        stride_list = None
        if os == 8:
            # 控制block2,block3,block4的第一个bottleneck的3x3卷积的stride
            # 这里指将block2内的第一个bottleneck的3x3卷集的stride设置为2
            # 这里指将block3内的第一个bottleneck的3x3卷集的stride设置为1
            # 这里指将block4内的第一个bottleneck的3x3卷集的stride设置为1
            stride_list = [2, 1, 1]
        elif os == 16:
            stride_list = [2, 2, 1]
        else:
            raise ValueError('resnet_atrous.py: output stride=%d is not supported.' % os)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # resnet的 block1
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        # resnet的 block2
        self.layer2 = self._make_layer(block, 64 * block.expansion, 128, layers[1], stride=stride_list[0])
        # resnet的 block3
        self.layer3 = self._make_layer(block, 128 * block.expansion, 256, layers[2], stride=stride_list[1], atrous=16 // os)
        # resnet的 block4,block4的atrous为列表，里面使用了multi-grid技术
        self.layer4 = self._make_layer(block, 256 * block.expansion, 512, layers[3], stride=stride_list[2],
                                       atrous=[item * 16 // os for item in atrous])
        self.layer5 = self._make_layer(block, 512 * block.expansion, 512, layers[3], stride=1, atrous=[item*16//os for item in atrous])
        self.layer6 = self._make_layer(block, 512 * block.expansion, 512, layers[3], stride=1, atrous=[item*16//os for item in atrous])
        self.layers = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_chans, out_chans, blocks, stride=1, atrous=None):
        downsample = None
        if atrous == None:
            # 当没有设置atrous,blocks=3时，atrous=[1,1,1]
            # 此时表示resnet的block1,或者block2,或者block3,或者block4内的bottleneck中的3x3卷积的膨胀系数为1，
            # 膨胀系数为1，就表示没有膨胀，还是标准卷积。
            atrous = [1] * blocks
        elif isinstance(atrous, int):
            # 当设置atrous=2,blocks=3时，atrous=[2,2,2]
            # 此时表示resnet的block1,或者block2,或者block3,或者block4内的bottleneck中的3x3卷积的膨胀系数为2
            atrous_list = [atrous] * blocks
            atrous = atrous_list
        # 如果atrous不是None,也不是一个整数，那么atrous被直接设定为[1,2,3]
        # 此时表示resnet的block1,或者block2,或者block3,或者block4内的bottleneck中的3个3x3卷积的膨胀系数分别为[1,2,3]
        
        if stride != 1 or in_chans != out_chans * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_chans, out_chans * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chans * block.expansion),
            )

        layers = []
        layers.append(block(in_chans, out_chans, stride=stride, atrous=atrous[0], downsample=downsample))
        in_chans = out_chans * block.expansion
        for i in range(1, blocks):
            layers.append(block(in_chans, out_chans, stride=1, atrous=atrous[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        layers_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        # 此时x为4倍下采样
        layers_list.append(x)
        x = self.layer2(x)
        # 此时x为8倍下采样
        layers_list.append(x)
        x = self.layer3(x)
        # 此时x为8倍或者16倍下采样，由本代码的123,125行的 stride_list决定
        # stride_list[2,1,1]时，就是8倍下采样
        # stride_list[2,2,1]时，就是16倍下采样
        
        layers_list.append(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        # 此时x为8倍或者16倍下采样，由本代码的123,125行的 stride_list决定
        # stride_list[2,1,1]时，就是8倍下采样
        # stride_list[2,2,1]时，就是16倍下采样
        layers_list.append(x)
        # return 4个feature map,分别是block1,block2,block3,block6的feature map
        return layers_list

def resnet34_atrous(pretrained=True, os=16, **kwargs):
    """Constructs a atrous ResNet-34 model."""
    model = ResNet_Atrous(BasicBlock, [3, 4, 6, 3], atrous=[1, 2, 1], os=os, **kwargs)
    if pretrained:
        old_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


def resnet50_atrous(pretrained=True, os=16, **kwargs):
    """Constructs a atrous ResNet-50 model."""
    model = ResNet_Atrous(Bottleneck, [3, 4, 6, 3], atrous=[1, 2, 1], os=os, **kwargs)
    if pretrained:
        old_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


def resnet101_atrous(pretrained=True, os=16, **kwargs):
    """Constructs a atrous ResNet-101 model."""
    model = ResNet_Atrous(Bottleneck, [3, 4, 23, 3], atrous=[1, 2, 1], os=os, **kwargs)
    if pretrained:
        old_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


from aspp import ASPP

class Config(object):
    # 决定本代码的123,125行的 stride_list的取值
    OUTPUT_STRIDE = 16
    #设定ASPP模块输出的channel数
    ASPP_OUTDIM = 256
    # Decoder中，shortcut的1x1卷积的channel数目
    SHORTCUT_DIM = 48
    # Decoder中，shortcut的卷积的核大小
    SHORTCUT_KERNEL = 1
    # 每个像素要被分类的类别数
    NUM_CLASSES = 21

class DeeplabV3Plus(nn.Module):
    def __init__(self, cfg, backbone=resnet50_atrous):
        super(DeeplabV3Plus, self).__init__()
        self.backbone = backbone(pretrained=False, os=cfg.OUTPUT_STRIDE)
        input_channel = 512 * self.backbone.block.expansion
        self.aspp = ASPP(in_chans=input_channel, out_chans=cfg.ASPP_OUTDIM, rate=16//cfg.OUTPUT_STRIDE)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.OUTPUT_STRIDE//4)

        indim = 64 * self.backbone.block.expansion
        self.shortcut_conv = nn.Sequential(
                nn.Conv2d(indim, cfg.SHORTCUT_DIM, cfg.SHORTCUT_KERNEL, 1, padding=cfg.SHORTCUT_KERNEL//2,bias=False),
                nn.BatchNorm2d(cfg.SHORTCUT_DIM),
                nn.ReLU(inplace=True),
        )
        self.cat_conv = nn.Sequential(
                nn.Conv2d(cfg.ASPP_OUTDIM+cfg.SHORTCUT_DIM, cfg.ASPP_OUTDIM, 3, 1, padding=1,bias=False),
                nn.BatchNorm2d(cfg.ASPP_OUTDIM),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv2d(cfg.ASPP_OUTDIM, cfg.ASPP_OUTDIM, 3, 1, padding=1, bias=False),
                nn.BatchNorm2d(cfg.ASPP_OUTDIM),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(cfg.ASPP_OUTDIM, cfg.NUM_CLASSES, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 利用backbone生成block1,2,3,4,5,6,7的feature maps
        layers = self.backbone(x)
        # layers[-1]是block7输出的feature map相对于原图下采样了16倍
        # 把block7的输出送入aspp
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        # 双线行插值上采样4倍
        feature_aspp = self.upsample_sub(feature_aspp)

        # layers[0],是block1输出的featuremap，相对于原图下采样的4倍，我们将它送入1x1x48的卷积中
        feature_shallow = self.shortcut_conv(layers[0])
        # aspp上采样4倍，变成相对于原图下采样4倍，与featue _shallow 拼接融合
        feature_cat = torch.cat([feature_aspp, feature_shallow],1)
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        result = self.upsample4(result)
        return result


cfg = Config()
model = DeeplabV3Plus(cfg, backbone=resnet50_atrous)
x = torch.randn((2, 3, 128, 128), dtype=torch.float32)
y = model(x)
print(y.shape)
