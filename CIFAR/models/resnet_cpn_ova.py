import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict


# ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


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

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, thres_type='multi', logit_temperature=0.01, global_thres=150):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.centers = nn.Parameter(torch.rand(num_classes, 512 * block.expansion))
        if thres_type == 'multi':
            self.rejection_threshold = nn.Parameter(torch.ones(1, num_classes))
        elif thres_type == 'one':
            self.rejection_threshold = nn.Parameter(torch.tensor(1.00))
        elif thres_type == 'const':
            self.rejection_threshold = torch.tensor(global_thres)        
        self.temperature_scale = torch.tensor(logit_temperature)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def disatance(self, features, centers):
        f_2 = features.pow(2).sum(dim=1, keepdim=True)
        c_2 = centers.pow(2).sum(dim=1, keepdim=True)
        dist = f_2 - 2*torch.matmul(features, centers.transpose(0,1)) + c_2.transpose(0,1)
        return dist

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        features = out.view(out.size(0), -1)
        out = self.disatance(features, self.centers)
        return out, features, self.centers

    def get_final_feature(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def ResNet18_cpn(num_classes=10, thres_type='multi', logit_temperature=0.01, global_thres=150):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, thres_type=thres_type, logit_temperature=logit_temperature, global_thres=global_thres)


def ResNet34_cpn(num_classes=10, thres_type='multi', logit_temperature=0.01, global_thres=150):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, thres_type=thres_type, logit_temperature=logit_temperature, global_thres=global_thres)


def ResNet50_cpn(num_classes=10, thres_type='multi', logit_temperature=0.01, global_thres=150):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, thres_type=thres_type, logit_temperature=logit_temperature, global_thres=global_thres)


def ResNet101_cpn(num_classes=10, thres_type='multi', logit_temperature=0.01, global_thres=150):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, thres_type=thres_type, logit_temperature=logit_temperature, global_thres=global_thres)


def ResNet152_cpn(num_classes=10, thres_type='multi', logit_temperature=0.01, global_thres=150):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, thres_type=thres_type, logit_temperature=logit_temperature, global_thres=global_thres)


def test():
    net = ResNet18_cpn()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
