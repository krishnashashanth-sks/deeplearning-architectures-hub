import torch.nn as nn
from layers import CustomDenseLayer,MultiBranchBlock,SimpleAttention

class AltasModel(nn.Module):
  def __init__(self,input_dim,hidden_dim,output_dim):
    super(AltasModel,self).__init__()
    self.initial_layer=CustomDenseLayer(input_dim,hidden_dim)
    self.multi_branch_block=MultiBranchBlock(hidden_dim,hidden_dim//2,hidden_dim//2)
    self.attention_mechanism=SimpleAttention(hidden_dim)
    self.final_output_layer=nn.Linear(hidden_dim,output_dim)
  def forward(self,x):
    x=self.initial_layer(x)
    x_branch=self.multi_branch_block(x)
    attended_features,_=self.attention_mechanism(x_branch,x_branch,x_branch)
    combined_features=attended_features+x_branch
    return self.final_output_layer(combined_features)
