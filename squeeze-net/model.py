import torch.nn as nn
from layers import FireModule
import torch

class SqueezeNet(nn.Module):
  def __init__(self,num_classes=1000):
    super(SqueezeNet,self).__init__()
    self.num_classes=num_classes
    self.features=nn.Sequential(
        nn.Conv2d(3,64,kernel_size=3,stride=2),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True), # Corrected stride to 2 for v1.1
        FireModule(64,16,64,64),
        FireModule(128,16,64,64),
        nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True),
        FireModule(128,32,128,128), # Adjusted channels to increase depth to 256
        FireModule(256,32,128,128),
        nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True),
        FireModule(256,48,192,192),
        FireModule(384,48,192,192),
        FireModule(384,64,256,256),
        FireModule(512,64,256,256)
    )
    self.classifier=nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Conv2d(512,self.num_classes,kernel_size=1),
        nn.ReLU(True),
        nn.AdaptiveAvgPool2d((1,1))
    )
    for m in self.modules():
      if isinstance(m,nn.Conv2d):
        if m.bias is not None:
          nn.init.constant_(m.bias,0)
  def forward(self,x):
    x=self.features(x)
    x=self.classifier(x)
    return torch.flatten(x,1)