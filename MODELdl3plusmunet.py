import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

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

class MobileUNet(nn.Module):
    def __init__(self, num_classes=4):
        super(MobileUNet, self).__init__()

        mobilenet = mobilenet_v2(pretrained=True)
        self.encoder = mobilenet.features

        self.aspp = ASPP(1280, 256)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 96, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.final_conv = nn.Conv2d(num_classes, num_classes, kernel_size=1)

    def forward(self, x):
        encoded = self.encoder(x)
        x = self.aspp(encoded)
        decoded = self.decoder(x)
        final_output = self.final_conv(decoded)
        final_output = F.interpolate(final_output, size=(200, 200), mode='bilinear', align_corners=True)
        return final_output

def get_model(num_classes):
    return MobileUNet(num_classes=num_classes)


