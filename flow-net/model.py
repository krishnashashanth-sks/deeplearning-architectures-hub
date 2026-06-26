import torch.nn as nn
import torch.nn.functional as F
import torch

class FlowNet(nn.Module):
  def __init__(self, input_dim: int, hidden_dim: int):
    super().__init__()
    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, 1) # Output a single scalar for logF(s)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

class ForwardPolicy(nn.Module):
  def __init__(self,input_dim:int,hidden_dim:int):
    super().__init__()
    self.fc1=nn.Linear(input_dim,hidden_dim)
    self.fc2=nn.Linear(hidden_dim,hidden_dim)
    self.fc3=nn.Linear(hidden_dim,2)
  def forward(self,x:torch.Tensor)->torch.Tensor:
    return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))