import torch.nn as nn
from layers import CustomBackbone,YOCONetworkHead

class YOCONetwork(nn.Module):
    def __init__(self, in_channels=3, num_anchors=3, num_classes=80):
        super(YOCONetwork, self).__init__()
        self.backbone = CustomBackbone(in_channels)
        backbone_out_channels = 1024 # Based on the CustomBackbone definition
        self.head = YOCONetworkHead(backbone_out_channels, num_anchors, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions