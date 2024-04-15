import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.DualBN import DualBN2d
from .utils.conv2d_dualMask import conv2d_dualMask

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use2BN=False, dualmask=False):
        super(BasicBlock, self).__init__()
        self.use2BN = use2BN
        self.dualmask = dualmask
        if self.use2BN:
            Norm2d = DualBN2d
        else:
            Norm2d = nn.BatchNorm2d
        if self.dualmask:
            conv2d = conv2d_dualMask
        else:
            conv2d = nn.Conv2d
        self.conv1 = conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = Norm2d(planes)
        self.conv2 = conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = Norm2d(planes)

        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion * planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(self.expansion * planes)
        #     )
        if stride != 1 or in_planes != self.expansion * planes:
            self.mismatch = True
            self.conv_sc = conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            self.bn_sc = Norm2d(self.expansion * planes)
        else:
            self.mismatch = False

    def forward(self, input):
        x, features, idx2BN, mask_type = input
        if self.use2BN:
            if self.dualmask:
                out = F.relu(self.bn1(self.conv1(x, mask_type), idx2BN))
            else:
                out = F.relu(self.bn1(self.conv1(x), idx2BN))
        else:
            if self.dualmask:
                out = F.relu(self.bn1(self.conv1(x, mask_type)))
            else:
                out = F.relu(self.bn1(self.conv1(x)))
        features.append(out)
        if self.use2BN:
            if self.dualmask:
                out = self.bn2(self.conv2(out, mask_type), idx2BN)
            else:
                out = self.bn2(self.conv2(out), idx2BN)                
        else:
            if self.dualmask:
                out = self.bn2(self.conv2(out, mask_type))
            else:
                out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        if self.mismatch:
            if self.use2BN:
                if self.dualmask:
                    out += self.bn_sc(self.conv_sc(x, mask_type), idx2BN)
                else:
                    out += self.bn_sc(self.conv_sc(x), idx2BN)
            else:
                if self.dualmask:
                    out += self.bn_sc(self.conv_sc(x, mask_type))
                else:
                    out += self.bn_sc(self.conv_sc(x))
        preact = out
        out = F.relu(out)
        features.append(out)
        return (out, features, idx2BN, mask_type)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, input):
        x, features = input
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        return (out, features)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, zero_init_residual=False, use2BN=False):
        super(ResNet, self).__init__()
        self.use2BN = use2BN
        self.in_planes = 64
        if num_classes == 1000:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = conv2d_dualMask(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use2BN:
            Norm2d = DualBN2d
        else:
            Norm2d = nn.BatchNorm2d
        self.bn1 = Norm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use2BN=use2BN, dualmask=True)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use2BN=use2BN, dualmask=True)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use2BN=use2BN, dualmask=True)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, use2BN=use2BN, dualmask=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, conv2d_dualMask):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        feat_m.append(self.layer4)
        return feat_m

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            raise NotImplementedError('ResNet unknown block error !!!')

        return [bn1, bn2, bn3, bn4]

    def _make_layer(self, block, planes, num_blocks, stride, use2BN=False, dualmask=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride, use2BN=use2BN, dualmask=dualmask))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, features=[], is_feat=False, idx2BN=None, mask_type=None):
        if self.use2BN:
            out = F.relu(self.bn1(self.conv1(x, mask_type), idx2BN))
        else:
            out = F.relu(self.bn1(self.conv1(x, mask_type)))
            # out = F.relu(self.bn1(self.conv1(x)))
        features.append(out)
        f0 = out
        if x.shape[-1] == 224: #imagenet
            out = self.maxpool(out)
        out, features, idx2BN, mask_type = self.layer1((out, features, idx2BN, mask_type))
        f1= out
        out, features, idx2BN, mask_type = self.layer2((out, features, idx2BN, mask_type))
        f2 = out
        out, features, idx2BN, mask_type = self.layer3((out, features, idx2BN, mask_type))
        f3 = out
        out, features, idx2BN, mask_type = self.layer4((out, features, idx2BN, mask_type))
        f4 = out
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        f5 = out
        out = self.linear(out)
        if is_feat:
            return [f0, f1, f2, f3, f4, f5], out, features
        else:
            return out, features


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)