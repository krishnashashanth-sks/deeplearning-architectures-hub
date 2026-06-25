import torch
import torch.nn as nn
import torch.nn.functional as F

class FNO2d(nn.Module):
  def __init__(self,in_channels,out_channels,modes1,modes2,width):
    super(FNO2d,self).__init__()
    self.in_channels=in_channels
    self.out_channels=out_channels
    self.modes1=modes1
    self.modes2=modes2
    self.width=width
    self.padding=9
    self.p=nn.Linear(in_channels,self.width)
    self.weights1=nn.Parameter(torch.view_as_real(torch.randn(self.width,self.width,self.modes1,self.modes2,dtype=torch.cfloat)))
    self.weights2=nn.Parameter(torch.view_as_real(torch.randn(self.width,self.width,self.modes1,self.modes2,dtype=torch.cfloat)))
    self.w=nn.Conv2d(self.width,self.width,1)
    self.w2=nn.Conv2d(self.width,self.width,1)
    self.fc1=nn.Linear(self.width,128)
    self.fc2=nn.Linear(128,out_channels)
  def forward(self,x):
    x=self.p(x.permute(0,2,3,1)).permute(0,3,1,2)
    x_padded=F.pad(x,[0,self.padding,0,self.padding])
    x_fft=torch.fft.rfft2(x_padded)
    out_fft=torch.zeros_like(x_fft)
    weights1_complex=torch.view_as_complex(self.weights1)
    weights2_complex=torch.view_as_complex(self.weights2)
    x_fft_modes1=x_fft[:,:,:self.modes1,:self.modes2]
    x_fft_modes2=x_fft[:,:,-self.modes1:,:self.modes2]
    out_fft[:,:,:self.modes1,:self.modes2]=torch.einsum('bixy,ioxy->boxy',x_fft_modes1,weights1_complex)
    out_fft[:,:,-self.modes1:,:self.modes2]=torch.einsum('bixy,ioxy->boxy',x_fft_modes2,weights2_complex)
    x_filtered=torch.fft.irfft2(out_fft,s=(x_padded.size(2),x_padded.size(3)))
    x_filtered=x_filtered[...,:-self.padding,:-self.padding]
    x=self.w(x)+x_filtered
    x=F.gelu(x)
    x=self.fc1(x.permute(0,2,3,1))
    x=F.gelu(x)
    x=self.fc2(x).permute(0,3,1,2)
    return x