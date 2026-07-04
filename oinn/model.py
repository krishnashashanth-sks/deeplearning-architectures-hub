import torch.nn as nn
from layers import LTSTALayer
import torch

class LTSTANet(nn.Module):
  def __init__(self,input_dim,output_dim,n_layers):
    super(LTSTANet,self).__init__()
    self.n_layers=n_layers
    self.input_dim=input_dim
    self.output_dim=output_dim
    self.layers=nn.ModuleList([
        LTSTALayer(input_dim,output_dim)for _ in range(n_layers)
    ])
  def forward(self,y):
    x_k=torch.zeros(y.size(0),self.output_dim,device=y.device)
    for i in range(self.n_layers):
      x_k=self.layers[i](x_k,y)
    return x_k