import torch
import torch.nn as nn


class CBSModule(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1, ratio=1):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(ratio * output_channel)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        out = self.silu(x)
        return out


class SPPFModule(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        # shape: out = (i+2*0-1)/1 + 1 = i
        self.conv = CBSModule(input_channel, output_channel, 1, 1, 0)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.pool3 = nn.MaxPool2d(3, 2)
        self.conv1 = CBSModule(output_channel, output_channel, 1, 1, 0)

    def forward(self, x):
        x = self.conv(x)
        p1 = self.pool1(x)
        p2 = self.pool2(p1)
        p3 = self.pool3(p2)
        x = torch.concat([x, p1, p2, p3], dim=1)
        out = self.conv1(x)
        return out


class DarknetBottleneck(nn.Module):
    def __init__(self, input_channel, output_channel, add=False):
        super().__init__()
        self.cbs1 = CBSModule(output_channel, 0.5 * output_channel, kernel_size=3, stride=1, padding=1)
        self.cbs2 = CBSModule(0.5 * output_channel, output_channel, 3, 1, 1)
        self.add = add

    def forward(self, x):
        x = self.cbs1(x)
        x1 = self.cbs2(x)
        if self.add:
            out = x + x1
        else:
            out = x1
        return out


class CSPLayerModule(nn.Module):
    def __init__(self, input_channel, output_channel, db_n):
        super().__init__()
        self.oc = output_channel
        self.cbs1 = CBSModule(input_channel, output_channel, 1, 1, 0)
        self.dbns = []
        for i in range(db_n):
            self.dbns.append(DarknetBottleneck(output_channel, output_channel*0.5))
        self.cbs2 = CBSModule(0.5*(db_n+2)*output_channel, output_channel, 1, 1, 0)

    def forward(self, x):
        x = self.cbs1(x)
        outputs = torch.split(x, 0.5*self.oc, dim=2)
        assert len(outputs) == 2
        for dbn in self.dbns:
            x = dbn(x)
            outputs.append(x)
        output = torch.concat(outputs)
        return output


class DecoupledHead(nn.Module):
    def __init__(self, input_channel, output_channel, reg_max=16):
        super().__init__()
        # box loss part
        self.cbs_box1 = CBSModule(input_channel, output_channel)
        self.cbs_box2 = CBSModule(output_channel, output_channel)
        self.conv_box = nn.Conv2d(output_channel, 4*reg_max, 1, 1, 0)
        # === box loss ===
        # cls loss part
        self.cbs_cls1 = CBSModule(input_channel, output_channel)
        self.cbs_cls2 = CBSModule(output_channel, output_channel)
        self.conv_cls = nn.Conv2d(output_channel, 4*reg_max, 1, 1, 0)
        # === cls loss ===

    def forward(self):
        pass



















