import torch
import torch.nn as nn


class BaseConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, padding, stride=1):
        super(BaseConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class InceptionV3(nn.Module):
    def __init__(self, in1x1, in3xn_1, in7xn_1, in7xn_2, out1x1, out3x3_1, out7xn_1, out7xn_2):
        super(InceptionV3, self).__init__()
        self.branch1x1 = BaseConvBlock(in1x1, out1x1, kernel=1, padding=0)
        self.branch3xn = nn.Sequential(
            BaseConvBlock(in1x1, out1x1, kernel=1, padding=0),
            BaseConvBlock(out1x1, in3xn_1, kernel=(1, 3), padding=(1, 1, 0, 0)),
            BaseConvBlock(in3xn_1, out3x3_1, kernel=(3, 1), padding=(0, 0, 1, 1))
        )
        self.branch7xn = nn.Sequential(
            BaseConvBlock(in1x1, out1x1, kernel=1, padding=0),
            BaseConvBlock(out1x1, in7xn_1, kernel=(1, 5), padding=(2, 2, 0, 0)),
            BaseConvBlock(in7xn_1, out7xn_1, kernel=(5, 1), padding=(0, 0, 2, 2)),
            BaseConvBlock(out7xn_1, in7xn_2, kernel=(1, 7), padding=(3, 3, 0, 0)),
            BaseConvBlock(in7xn_2, out7xn_2, kernel=(7, 1), padding=(0, 0, 3, 3))
        )
        self.branchPool = nn.Sequential(
            nn.AvgPool2d(in1x1),
            BaseConvBlock(in1x1, out1x1, kernel=1, padding=0)
        )

    def forward(self, x):
        x1 = self.branch1x1(x)
        x2 = self.branch3xn(x)
        x3 = self.branch7xn(x)
        x4 = self.branchPool(x)
        out = torch.cat((x1, x2, x3, x4), dim=1)
        return out






