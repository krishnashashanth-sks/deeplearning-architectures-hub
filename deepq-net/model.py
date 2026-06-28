import torch.nn as nn

class DQNetwork(nn.Module):
  def __init__(self,state_dim,action_dim,hidden_size=128):
    super(DQNetwork,self).__init__()
    self.fc1=nn.Linear(state_dim,hidden_size)
    self.relu=nn.ReLU()
    self.fc2=nn.Linear(hidden_size,action_dim)
  def forward(self,state):
    return self.fc2(self.relu(self.fc1(state)))