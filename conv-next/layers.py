import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
  def __init__(self,dim,drop_path=0.,layer_scale_init_value=1e-6):
    super().__init__()
    self.dwconv=nn.Conv2d(dim,dim,kernel_size=7,padding=3,groups=dim)
    self.norm=nn.LayerNorm(dim,eps=1e-6)
    self.pwconv1=nn.Linear(dim,4*dim)
    self.act=nn.GELU()
    self.pwconv2=nn.Linear(4*dim,dim)
    self.gamma=nn.Parameter(layer_scale_init_value*torch.ones((dim)),requires_grad=True)if layer_scale_init_value >0 else None
    self.drop_path=nn.Identity() # Placeholder for stochastic depth

  def forward(self,x):
    identity=x
    x=self.dwconv(x)
    x=x.permute(0,2,3,1) # (N, C, H, W) -> (N, H, W, C)
    x=self.norm(x)
    x=self.pwconv1(x)
    x=self.act(x)
    x=self.pwconv2(x)
    x=x.permute(0,3,1,2) # (N, H, W, C) -> (N, C, H, W)
    if self.gamma is not None:
      x=self.gamma.unsqueeze(-1).unsqueeze(-1)*x # Reshape gamma for broadcasting
    return identity+self.drop_path(x)
  
class ConvNeXtDownSample(nn.Module):
  def __init__(self,in_channels,out_channels):
    super().__init__()
    self.norm=nn.LayerNorm(in_channels,eps=1e-6)
    self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=2,stride=2)
  def forward(self,x):
    x=x.permute(0,2,3,1)
    x=self.norm(x)
    x=x.permute(0,3,1,2)
    return self.conv(x)
  
class LayerNormForConvNext(nn.Module):
      def __init__(self,normalized_shape,eps=1e-6):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(normalized_shape))
        self.bias=nn.Parameter(torch.zeros(normalized_shape))
        self.eps=eps
      def forward(self,x):
        u=x.mean(1,keepdim=True)
        s=(x-u).pow(2).mean(1,keepdim=True)
        x=(x-u)/torch.sqrt(s+self.eps)
        x=self.weight[:,None,None]*x+self.bias[:,None,None]
        return x