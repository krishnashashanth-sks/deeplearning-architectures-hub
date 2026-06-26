from utils import *
import torch

def generate_bit_string(forward_policy, N: int) -> list: 
  """
  Generates a complete bit string using the trained forward policy.

  Args:
    forward_policy: The GFlowNet's forward policy network.
    N: The desired length of the bit string.

  Returns:
    A list representing the generated bit string (e.g., [0, 1, 0, 1, 1, 0, 1, 0, 0, 1]).
  """
  current_state = []
  for _ in range(N):
    state_tensor = start_to_tensor(current_state).unsqueeze(0)
    with torch.no_grad():
      action_logits = forward_policy(state_tensor)

    # Sample action from the policy distribution
    action_distribution = torch.distributions.Categorical(logits=action_logits)
    action = action_distribution.sample().item()
    current_state.append(action)
  return current_state
