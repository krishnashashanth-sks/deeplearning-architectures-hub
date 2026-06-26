import torch
import torch.distributions

def sample_trajectory(forward_policy, N: int, reward_function, target_ones: int = 5) -> dict:
  """
  Samples a complete trajectory (bit string) using the forward policy.

  Args:
    forward_policy: The GFlowNet's forward policy network.
    N: The desired length of the bit string.
    reward_function: A function that computes the reward for a final bit string.
    target_ones: The target number of ones for the reward function.

  Returns:
    A dictionary containing:
      - "states": A list of states (partial bit strings) encountered.
      - "actions": A list of actions taken (0 or 1).
      - "reward": The reward of the final bit string.
  """
  current_state = []  # Represents the partial bit string
  states = []         # To store all intermediate states
  actions = []        # To store all actions taken

  for _ in range(N):
    states.append(list(current_state))  # Store a copy of the current state
    state_tensor = start_to_tensor(current_state).unsqueeze(0)  # Convert to tensor for policy input

    with torch.no_grad():  # Do not track gradients during sampling
      action_logits = forward_policy(state_tensor)

    # Sample action from the policy distribution
    action_distribution = torch.distributions.Categorical(logits=action_logits)
    action = action_distribution.sample().item()

    current_state.append(action) # Update the current state
    actions.append(action)      # Store the action

  # Calculate the reward for the final, complete bit string
  final_reward = reward_function(current_state, target_ones)

  return {
      "states": states,
      "actions": actions,
      "reward": final_reward
  }

def reward_function(bit_string:list,target_ones:int=5)->float:
  num_ones=sum(bit_string)
  if num_ones==target_ones:
    return 1.0
  else:
    return 0.01


def start_to_tensor(state: list,N=10) -> torch.Tensor:
  """
  Converts a partial bit string (list of ints) into a padded tensor.
  Padding is done with -1 to distinguish from 0/1 bits.
  """
  tensor_state = torch.tensor(state, dtype=torch.float32)
  if len(tensor_state) < N:
    # Use -1 for padding to clearly distinguish from 0 or 1 bits
    padding = torch.ones(N - len(tensor_state), dtype=torch.float32) * -1
    tensor_state = torch.cat([tensor_state, padding])
  return tensor_state