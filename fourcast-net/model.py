import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import FNO2d

class FourCastNet(nn.Module):
  def __init__(self,in_channels,out_channels,modes1,modes2,width,num_fno_blocks=1):
    super(FourCastNet,self).__init__()
    self.in_channels=in_channels
    self.out_channels=out_channels
    self.modes1=modes1
    self.modes2=modes2 # Corrected from self.modes2=modes2,
    self.width=width
    self.num_fno_blocks=num_fno_blocks
    self.initial_proj=nn.Conv2d(in_channels,self.width,kernel_size=1)
    self.encoder1_downsample=nn.Conv2d(self.width,self.width,kernel_size=4,stride=2,padding=1)
    self.encoder1_fno_blocks=nn.ModuleList([
        FNO2d(self.width,self.width,modes1//2,modes2//2,self.width)for _ in range(num_fno_blocks)
    ])
    self.encoder2_downsample = nn.Conv2d(self.width, self.width, kernel_size=4, stride=2, padding=1)
    self.encoder2_fno_blocks = nn.ModuleList([
            FNO2d(self.width, self.width, modes1 // 2, modes2 // 2, self.width) for _ in range(num_fno_blocks)
        ])
    self.bottleneck_fno_blocks=nn.ModuleList([
        FNO2d(self.width,self.width,modes1//4,modes2//4,self.width)for _ in range(num_fno_blocks)
    ])

    self.decoder1_upsample=nn.ConvTranspose2d(self.width,self.width,kernel_size=4,stride=2,padding=1)
    # Initialize decoder1_fno_blocks with correct input channels
    decoder1_fno_list = []
    decoder1_fno_list.append(FNO2d(2 * self.width, self.width, modes1 // 2, modes2 // 2, self.width))
    for _ in range(self.num_fno_blocks - 1):
        decoder1_fno_list.append(FNO2d(self.width, self.width, modes1 // 2, modes2 // 2, self.width))
    self.decoder1_fno_blocks = nn.ModuleList(decoder1_fno_list)

    self.decoder2_upsample=nn.ConvTranspose2d(self.width,self.width,kernel_size=4,stride=2,padding=1)
    # Initialize decoder2_fno_blocks with correct input channels
    decoder2_fno_list = []
    decoder2_fno_list.append(FNO2d(2 * self.width, self.width, modes1, modes2, self.width))
    for _ in range(self.num_fno_blocks - 1):
        decoder2_fno_list.append(FNO2d(self.width, self.width, modes1, modes2, self.width))
    self.decoder2_fno_blocks = nn.ModuleList(decoder2_fno_list)

    self.final_proj=nn.Conv2d(self.width,out_channels,kernel_size=1)
  def forward(self,x):
    x=self.initial_proj(x)
    skip1=x
    x=self.encoder1_downsample(x)
    for fno_block in self.encoder1_fno_blocks:
      x=fno_block(x)
    skip2=x
    x=self.encoder2_downsample(x)
    for fno_block in self.encoder2_fno_blocks:
      x=fno_block(x)
    for fno_block in self.bottleneck_fno_blocks:
      x=fno_block(x)
    for fno_block in self.bottleneck_fno_blocks:
      x=fno_block(x)
    x=self.decoder1_upsample(x)
    x=torch.cat([x,skip2],dim=1) # Changed dim=2 to dim=1
    for fno_block in self.decoder1_fno_blocks:
      x=fno_block(x)
    x=self.decoder2_upsample(x)
    x=torch.cat([x,skip1],dim=1) # Changed dim=2 to dim=1
    for fno_block in self.decoder2_fno_blocks:
      x=fno_block(x)
    return self.final_proj(x)