import torch
import torch.nn as nn

class ODEFunc(nn.Module):
  def __init__(self,input_dim,hidden_dim,output_dim):
    super(ODEFunc,self).__init__()
    self.net=nn.Sequential(
        nn.Linear(input_dim+1,hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim,hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim,output_dim)
    )
  def forward(self,t,x):
    if t.dim()==0:
      t_reshaped=t.expand(t.size(0),1)
    else:
      t_reshaped=t.unsqueeze(1).expand(-1,x.size(0),1).squeeze(1)
    input_nn=torch.cat((t_reshaped,x),dim=1)
    return self.net(input_nn)
  
class TrueODE(nn.Module):
    def forward(self, t, x):
        # x is a tensor of shape (batch_size, state_dim)
        # For a 2D spiral:
        # dx/dt = -0.1 * x[0] + 2 * x[1]
        # dy/dt = -2 * x[0] - 0.1 * x[1]
        dxdt = -0.1 * x[:, 0] + 2 * x[:, 1]
        dydt = -2 * x[:, 0] - 0.1 * x[:, 1]
        # The output should have the same shape as x
        return torch.stack([dxdt, dydt], dim=1)