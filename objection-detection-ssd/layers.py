import torch.nn as nn

class SSDHead(nn.Module):
  def __init__(self,in_channels,num_anchors,num_classes):
    super().__init__()
    self.num_anchors=num_anchors
    self.num_classes=num_classes
    self.cls_head=nn.Conv2d(in_channels,num_anchors*num_classes,kernel_size=3,padding=1)
    self.reg_head=nn.Conv2d(in_channels,num_anchors*4,kernel_size=3,padding=1)
  def forward(self,x):
    cls_preds=self.cls_head(x)
    reg_preds=self.reg_head(x)
    return cls_preds,reg_preds