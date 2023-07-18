from yolo.commons import *
import torch.nn as nn

model_params = {
    "n": [0.33, 0.25, 2],
    "s": [0.33, 0.5, 2],
    "m": [0.67, 0.75, 1.5],
    "l": [1, 1, 1],
    "x": [1, 1.25, 1]
}


class YOLOV8(nn.Module):
    def __init__(self, input_image, model_type):
        super().__init__()
        depth_multiple, width_multiple, ratio = model_params[model_type]
        b, h, w, c = input_image.shape()
        self.cbs1 = CBSModule(c, 64*width_multiple, 3, 2, 1)
        self.cbs2 = CBSModule(64*width_multiple, 128*width_multiple, 3, 2, 1)
        self.cps































