import torch
import torch.nn as nn


class BaseConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, padding=0):
        super(BaseConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class InceptionV2(nn.Module):
    def __init__(self, in1x1, in3x3, out1x1, out3x3, out3x3_middle):
        super(InceptionV2, self).__init__()
        self.branch1x1 = BaseConvBlock(in1x1, out1x1, kernel=1, padding=0)
        self.branch3x3 = nn.Sequential(
            BaseConvBlock(in1x1, in3x3, kernel=1, padding=0),
            BaseConvBlock(in3x3, out3x3, kernel=3, padding=1)
        )
        self.branch5x5 = nn.Sequential(
            BaseConvBlock(in1x1, in3x3, kernel=1, padding=0),
            BaseConvBlock(in3x3, out3x3_middle, kernel=3, padding=1),
            BaseConvBlock(out3x3_middle, out3x3, kernel=3, padding=1)
        )
        self.branchPool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BaseConvBlock(in1x1, out1x1, kernel=1, padding=0)
        )

    def forward(self, x):
        b1 = self.branch1x1(x)
        b2 = self.branch3x3(x)
        b3 = self.branch5x5(x)
        b4 = self.branchPool(x)
        out = torch.cat((b1, b2, b3, b4), dim=1)
        return out














