import torch.nn as nn
from layers import C4GConvLayer
from e2cnn import gspaces
from e2cnn import nn as enn

class ENNModel(nn.Module):
  def __init__(self,in_channels,num_classes):
    super(ENNModel,self).__init__()
    self.r2_act=gspaces.Rot2dOnR2(N=4) # Changed N=2 to N=4 to ensure consistent C4 equivariance
    self.conv1=C4GConvLayer(in_channels,16,kernel_size=5,padding=2)
    self.pool1=enn.PointwiseMaxPool(self.conv1.out_type,kernel_size=2,stride=2)
    self.conv2=enn.R2Conv(
        self.conv1.out_type,
        enn.FieldType(self.r2_act,32*[self.r2_act.regular_repr]),
        kernel_size=5,
        padding=2,
        bias=True
    )
    self.act2=enn.ReLU(self.conv2.out_type)
    self.pool2 = enn.PointwiseMaxPool(self.conv2.out_type, kernel_size=2, stride=2) # Corrected: Use enn.PointwiseMaxPool
    num_features_before_linear=self.pool2.out_type.size
    self.global_pool=nn.AdaptiveAvgPool2d(1)
    self.classifier=nn.Linear(num_features_before_linear,num_classes)
  def forward(self,x):
    x=self.pool2(self.act2(self.conv2(self.pool1(self.conv1(x)))))
    x_tensor=x.tensor
    x_pooled=self.global_pool(x_tensor)
    x_flattened=x_pooled.view(x_pooled.size(0),-1)
    return self.classifier(x_flattened)