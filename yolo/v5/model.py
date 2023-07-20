from yolo.commons import *


class YoloV5(nn.Module):
    def __init__(self, image):
        super().__init__()
        self.input_ = image
        self.cbs1 = CBSModule()