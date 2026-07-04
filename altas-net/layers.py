import torch.nn as nn
import torch

class CustomDenseLayer(nn.Module):
  def __init__(self,in_features,out_features):
    super(CustomDenseLayer,self).__init__()
    self.linear=nn.Linear(in_features,out_features)
    self.activation=nn.ReLU()
  def forward(self,x):
    return self.activation(self.linear(x))

class SimpleAttention(nn.Module):
  def __init__(self,feature_dim):
    super(SimpleAttention,self).__init__()
    self.query_transform=nn.Linear(feature_dim,feature_dim)
    self.key_transform=nn.Linear(feature_dim,feature_dim)
    self.value_transform=nn.Linear(feature_dim,feature_dim)
    self.softmax=nn.Softmax(dim=-1)
  def forward(self,query,keys,values):
    Q=self.query_transform(query)
    K=self.key_transform(keys) # Changed 'key' to 'keys'
    V=self.value_transform(values)
    attention_scores=torch.matmul(Q,K.transpose(-2,-1))/(Q.size(-1)**0.5) # Changed 'q' to 'Q'
    attention_weights=self.softmax(attention_scores)
    output=torch.matmul(attention_weights,V)
    return output,attention_weights

class MultiBranchBlock(nn.Module):
  def __init__(self,input_dim,branch1_out_dim,branch2_out_dim):
    super(MultiBranchBlock,self).__init__() # Corrected super() call
    self.branch1=nn.Sequential(
        CustomDenseLayer(input_dim,branch1_out_dim),
        nn.BatchNorm1d(branch1_out_dim) # Changed from BatchNorm2d to BatchNorm1d
    )
    self.branch2=nn.Sequential(
        nn.Linear(input_dim,branch2_out_dim),
        nn.ReLU()
    )
  def forward(self,x):
    output_branch1=self.branch1(x)
    output_branch2=self.branch2(x)
    return torch.cat((output_branch1,output_branch2),dim=1)