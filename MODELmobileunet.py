import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

class MobileUNet(nn.Module):
    def __init__(self, num_classes=4):
        super(MobileUNet, self).__init__()

        mobilenet = mobilenet_v2(pretrained=True)
        self.encoder = mobilenet.features

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 96, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.final_conv = nn.Conv2d(num_classes, num_classes, kernel_size=1)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        final_output = self.final_conv(decoded)
        final_output = F.interpolate(final_output, size=(200, 200), mode='bilinear', align_corners=True)
        return final_output
def get_model(num_classes):
    return MobileUNet(num_classes=num_classes)