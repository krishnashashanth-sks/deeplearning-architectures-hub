import torch.nn as nn

class ReceptanceWeightedKV(nn.Module):
  def __init__(self,input_dim,key_dim,value_dim,output_dim):
    super(ReceptanceWeightedKV,self).__init__()
    self.input_dim=input_dim
    self.key_dim=key_dim
    self.value_dim=value_dim
    self.output_dim=output_dim
    self.key_layer=nn.Linear(input_dim,key_dim)
    self.value_layer=nn.Linear(input_dim,value_dim)
    self.receptance_layer=nn.Linear(input_dim,value_dim)
    self.receptance_activation=nn.Sigmoid()
    self.output_layer=nn.Linear(value_dim,output_dim)

  def forward(self,x):
    key=self.key_layer(x) # Key is generated but not used in this basic feedforward model
    value=self.value_layer(x)
    receptance_raw=self.receptance_layer(x)
    receptance=self.receptance_activation(receptance_raw)
    weighted_value=receptance*value
    return self.output_layer(weighted_value)
