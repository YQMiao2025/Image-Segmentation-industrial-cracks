import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101
from torchvision.models.resnet import ResNet50_Weights, ResNet101_Weights

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )

        self.conv5 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x4 = self.relu(self.bn4(self.conv4(x)))
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.relu(self.bn5(self.conv5(x)))

        return x


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, backbone='resnet50'):
        super(DeepLabV3Plus, self).__init__()
        if backbone == 'resnet50':
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == 'resnet101':
            self.backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            in_channels = 2048
            low_level_channels = 256
        else:
            raise NotImplementedError

        self.aspp = ASPP(in_channels, 256)
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + low_level_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)
        )

        self._init_weight()

    def forward(self, x):
        x, low_level_feat = self.extract_features(x)
        x = self.aspp(x)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, size=(x.size(2) * 4, x.size(3) * 4), mode='bilinear', align_corners=True)

        return x

    def extract_features(self, x):
        low_level_feat = self.backbone.conv1(x)
        low_level_feat = self.backbone.bn1(low_level_feat)
        low_level_feat = self.backbone.relu(low_level_feat)
        low_level_feat = self.backbone.maxpool(low_level_feat)

        low_level_feat = self.backbone.layer1(low_level_feat)
        x = self.backbone.layer2(low_level_feat)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def get_model(num_classes, backbone='resnet50'):
    return DeepLabV3Plus(num_classes=num_classes, backbone=backbone)


