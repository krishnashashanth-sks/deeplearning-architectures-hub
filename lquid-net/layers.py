import torch.nn as nn
import torch

class DynamicsFunction(nn.Module):
  def __init__(self, hidden_size: int, input_size: int):
    super(DynamicsFunction, self).__init__()
    combined_input_size = hidden_size + input_size
    self.mlp = nn.Sequential(
        nn.Linear(combined_input_size, hidden_size * 2),
        nn.ReLU(),
        nn.Linear(hidden_size * 2, hidden_size)
    )
  def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    combined_hx = torch.cat((h, x), dim=-1)
    dh_dt = self.mlp(combined_hx)
    return dh_dt

class ODEFunc(nn.Module):
      def __init__(self,dynamics_func,x_sequence,t_points):
        super(ODEFunc,self).__init__()
        self.dynamics_func=dynamics_func
        self.x_sequence=x_sequence
        self.t_points=t_points

      def forward(self,t,h):
        # Find the closest input in x_sequence for the current time t
        time_idx = torch.argmin(torch.abs(self.t_points - t))
        current_x = self.x_sequence[:, time_idx, :]
        return self.dynamics_func(h,current_x)