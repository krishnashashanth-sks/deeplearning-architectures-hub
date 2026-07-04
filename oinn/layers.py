import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftThresholding(nn.Module):
  def __init__(self):
    super(SoftThresholding,self).__init__()
  def forward(self,z,tau):
    return torch.sign(z)*torch.relu(torch.abs(z)-tau)
  
class LTSTALayer(nn.Module):
  def __init__(self,input_dim,output_dim):
    super(LTSTALayer,self).__init__()
    self.R_k=nn.Parameter(torch.Tensor(output_dim,input_dim))
    self.S_k=nn.Parameter(torch.Tensor(output_dim,output_dim))
    self.theta_k=nn.Parameter(torch.Tensor(1))
    self.soft_threshold=SoftThresholding()
    self._initialize_parameters()
  def _initialize_parameters(self):
    nn.init.xavier_uniform_(self.R_k)
    nn.init.xavier_uniform_(self.S_k)
    nn.init.constant_(self.theta_k,0)
  def forward(self,x_k,y):
    term_R_y=F.linear(y,self.R_k)
    term_s_xk=F.linear(x_k,self.S_k)
    z_k_plus_1=term_s_xk+term_R_y
    x_k_plus_1=self.soft_threshold(z_k_plus_1,F.relu(self.theta_k))
    return x_k_plus_1