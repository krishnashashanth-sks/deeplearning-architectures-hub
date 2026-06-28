import random
import torch
from collections import deque

class ReplayBuffer:
  def __init__(self,capacity):
    self.buffer=deque(maxlen=capacity)
  def add(self,state,action,reward,next_state,done):
    state=state.tolist() if isinstance(state,np.ndarray) else state
    next_state=next_state.tolist() if isinstance(next_state,np.ndarray) else next_state
    self.buffer.append((state,action,reward,next_state,done))
  def sample(self,buffer_size):
    experiences=random.sample(self.buffer,buffer_size)
    states,actions,rewards,next_states,dones=zip(*experiences)
    states=torch.tensor(states,dtype=torch.float32)
    actions=torch.tensor(actions,dtype=torch.long).unsqueeze(-1)
    rewards=torch.tensor(rewards,dtype=torch.float32).unsqueeze(-1)
    next_states=torch.tensor(next_states,dtype=torch.float32)
    dones=torch.tensor(dones,dtype=torch.float32).unsqueeze(-1)
    return states,actions,rewards,next_states,dones
  def __len__(self):
    return len(self.buffer)