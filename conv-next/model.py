import torch
import torch.nn as nn
from layers import *

class ConvNeXt(nn.Module):
  def __init__(self,
               in_chans=3,
               num_classes=1000,
               depths=[3,3,9,3],
               dims=[96,192,384,768],
               drop_path_rate=0.,
               layer_scale_init_value=1e-6,
               head_init_scale=1.):
    super().__init__()
    self.num_classes=num_classes
    self.stem=nn.Sequential(
        nn.Conv2d(in_chans,dims[0],kernel_size=4,stride=4),
        LayerNormForConvNext(dims[0],eps=1e-6)
    )

    dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
    self.stages=nn.ModuleList()
    cur=0
    for i in range(len(depths)):
      if i>0:
        self.stages.append(
            ConvNeXtDownSample(in_channels=dims[i-1],out_channels=dims[i])
        )
      blocks=[]
      for j in range(depths[i]):
        blocks.append(
            ConvNeXtBlock(
                dim=dims[i],
                drop_path=dp_rates[cur+j],
                layer_scale_init_value=layer_scale_init_value
            )
        )
      self.stages.append(nn.Sequential(*blocks))
      cur+=depths[i]
    self.norm=nn.LayerNorm(dims[-1],eps=1e-6)
    self.avg_pool=nn.AdaptiveAvgPool2d(1)
    self.head=nn.Linear(dims[-1],num_classes)
    self.apply(self._init_weights)
    self.head.weight.data.mul_(head_init_scale)
    self.head.bias.data.mul_(head_init_scale)

  def _init_weights(self,m):
    if isinstance(m,(nn.Conv2d,nn.Linear)):
      nn.init.trunc_normal_(m.weight,std=.02)
      if m.bias is not None:
        nn.init.constant_(m.bias,0)

  def forward(self,x):
    x = self.stem(x)
    for stage in self.stages:
      x = stage(x)
    x = self.avg_pool(x).flatten(1)
    x = self.norm(x)
    x = self.head(x)
    return x