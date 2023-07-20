from yolo.commons import *
import torch.nn as nn

model_params = {
    "n": [0.33, 0.25, 2],
    "s": [0.33, 0.5, 2],
    "m": [0.67, 0.75, 1.5],
    "l": [1, 1, 1],
    "x": [1, 1.25, 1]
}


class YoloV8(nn.Module):
    def __init__(self, model_type, class_num, reg_max=16):
        super().__init__()
        depth_multiple, width_multiple, ratio = model_params[model_type]
        # Backbone
        self.cbs1 = CBSModule(3, 64*width_multiple, 3, 2, 1)
        self.stage_layer1 = nn.Sequential(
            CBSModule(64*width_multiple, 128*width_multiple, 3, 2, 1),
            C2fModule(128*width_multiple, 128*width_multiple, 3*depth_multiple, add=True)
        )
        self.stage_layer2 = nn.Sequential(
            CBSModule(128 * width_multiple, 256 * width_multiple, 3, 2, 1),
            C2fModule(256 * width_multiple, 256 * width_multiple, 6*depth_multiple, add=True)
        )
        self.stage_layer3 = nn.Sequential(
            CBSModule(256 * width_multiple, 512 * width_multiple, 3, 2, 1),
            C2fModule(512 * width_multiple, 512 * width_multiple, 6*depth_multiple, add=True)
        )
        self.stage_layer4 = nn.Sequential(
            CBSModule(512 * width_multiple, 512 * width_multiple * ratio, 3, 2, 1),
            C2fModule(512 * width_multiple * ratio, 512 * width_multiple * ratio, 3 * depth_multiple, add=True),
            SPPFModule(512 * width_multiple * ratio, 512 * width_multiple * ratio)
        )
        # TopDown
        self.upsample1 = nn.Upsample(2)
        # or self.upsample = nn.ConvTranspose2d(512 * width_multiple * ratio, 512 * width_multiple * ratio,
        # kernel_size=3, stride=2, padding=1) slower
        self.c2f1 = C2fModule(512 * width_multiple * (ratio+1), 512 * width_multiple, 3*depth_multiple, False)
        self.upsample2 = nn.Upsample(2)
        self.c2f2 = C2fModule(768 * width_multiple, 256 * width_multiple, 3*depth_multiple, False)
        # Down Sample
        self.cbs2 = CBSModule(256 * width_multiple, 256 * width_multiple, 3, 2, 1)
        self.c2f3 = C2fModule(768 * width_multiple, 512 * width_multiple, 3*depth_multiple, False)
        self.cbs3 = CBSModule(512 * width_multiple, 512 * width_multiple, 3, 2, 1)
        self.c2f4 = C2fModule(512 * width_multiple * (ratio+1), 512 * width_multiple * ratio, 3 * depth_multiple, False)
        self.decoupled_head1 = V8DecoupledHead(256 * width_multiple, 256 * width_multiple, class_num, reg_max)
        self.decoupled_head2 = V8DecoupledHead(512 * width_multiple, 512 * width_multiple, class_num, reg_max)
        self.decoupled_head3 = V8DecoupledHead(512 * width_multiple * ratio, 512 * width_multiple * ratio, class_num, reg_max)

    def forward(self, x):
        pass


























