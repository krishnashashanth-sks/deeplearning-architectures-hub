from layers import *
import torch.nn as nn
from torchdiffeq import odeint # Import odeint

class LiquidNeuralNetwork(nn.Module):
  def __init__(self,hidden_size:int,input_size:int,output_dim):
    super(LiquidNeuralNetwork,self).__init__()
    self.dynamics_func=DynamicsFunction(hidden_size,input_size)
    self.hidden_size=hidden_size
    self.input_size=input_size
    self.output_layer = nn.Linear(hidden_size, output_dim)

  def forward(self,h0:torch.Tensor,x_sequence:torch.Tensor,t:torch.Tensor)->torch.Tensor:
    ode_func=ODEFunc(self.dynamics_func,x_sequence,t)
    out=odeint(ode_func,h0,t,method='dopri5')
    h_sequence = self.lnn(h0, out, t)
    final_h = h_sequence[-1, :, :]
    output = self.output_layer(final_h)
    return output

