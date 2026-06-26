from train import train
from model import *
from utils import *
from losses import *
from inference import generate_bit_string

N=10

input_dim = N # N is the length of the bit string, defined earlier as 10
hidden_dim = 64 # A reasonable choice for a small toy problem

# Instantiate the networks
forward_policy = ForwardPolicy(input_dim, hidden_dim)
flow_network = FlowNet(input_dim, hidden_dim)

# Combine parameters for the optimizer
params = list(forward_policy.parameters()) + list(flow_network.parameters())

# Set up the optimizer
optimizer = torch.optim.Adam(params, lr=1e-3)


num_episodes = 2000
batch_size = 64
target_ones = 5 

losses,rewards=train(num_episodes,batch_size,N,forward_policy,reward_function,target_ones,optimizer,gfn_loss,flow_network)

print("\nGenerating 5 sample bit strings:")
for i in range(5):
  generated_string = generate_bit_string(forward_policy, N)
  num_ones = sum(generated_string)
  reward = reward_function(generated_string, target_ones)
  print(f"  Generated String {i+1}: {generated_string} (Ones: {num_ones}, Reward: {reward:.2f})")
