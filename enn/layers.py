import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as enn

class EquivariantLinear(nn.Module):
  def __init__(self,in_features,out_features,group_elements):
    super(EquivariantLinear,self).__init__()
    self.in_features=in_features
    self.out_features=out_features
    self.group_elementes=group_elements
    self.weight=nn.Parameter(torch.Tensor(out_features,in_features))
    self.bias=nn.Parameter(torch.Tensor(out_features))
    nn.init.kaiming_uniform_(self.weight,a=5**0.5)
    fan_in,_=nn.init._calculate_fan_in_and_fan_out(self.weight)
    bound=1/fan_in**0.5
    nn.init.uniform_(self.bias,-bound,bound)
  def forward(self,x):
    return torch.nn.functional.linear(x,self.weight,self.bias)
  
class C4GConvLayer(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size,padding=1):
    super(C4GConvLayer,self).__init__()
    self.r2_act=gspaces.Rot2dOnR2(N=4)
    self.in_type=enn.FieldType(self.r2_act,in_channels*[self.r2_act.trivial_repr])
    self.out_type=enn.FieldType(self.r2_act,out_channels*[self.r2_act.regular_repr])
    self.g_conv=enn.R2Conv(
        self.in_type,
        self.out_type,
        kernel_size=kernel_size,
        padding=padding,
        bias=True
    )
    self.activation=enn.ReLU(self.out_type)
  def forward(self,x):
    return self.activation(self.g_conv(enn.GeometricTensor(x,self.in_type)))