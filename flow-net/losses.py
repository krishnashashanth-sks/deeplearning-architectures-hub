import torch.nn.functional as F
import torch
from utils import *

def gfn_loss(forward_policy: torch.nn.Module, flow_network: torch.nn.Module, trajectories: list) -> torch.Tensor:
    """
    Computes the Trajectory Balance (TB) loss for a batch of trajectories.

    Args:
        forward_policy: The GFlowNet's forward policy network.
        flow_network: The GFlowNet's flow network, which outputs logF(s) for a state s.
        trajectories: A list of trajectory dictionaries, each containing 'states', 'actions', 'reward'.

    Returns:
        A scalar tensor representing the mean squared Trajectory Balance loss over the batch.
    """
    losses = []

    # Get log Z from the flow network at the start state (empty list)
    # The start_to_tensor function converts [] to a padded tensor.
    s0_tensor = start_to_tensor([]).unsqueeze(0) # Add batch dimension
    log_z = flow_network(s0_tensor).squeeze(-1) # Ensure it's a scalar value

    for trajectory_data in trajectories:
        states = trajectory_data['states']   # List of partial bit strings [s0, s1, ..., s_N-1]
        actions = trajectory_data['actions'] # List of actions taken [a0, a1, ..., a_N-1]
        reward = trajectory_data['reward']

        # Ensure reward is a tensor and add epsilon for log(0) safety
        log_reward = torch.log(torch.tensor(reward + 1e-8, dtype=torch.float32, device=log_z.device))

        # Sum of log P_F(s_{t+1}|s_t) over the trajectory
        sum_log_pf = torch.tensor(0.0, device=log_z.device)

        # Iterate through states and actions to accumulate log P_F
        for i in range(len(actions)):
            s_t = states[i] # Current partial state before action a_i
            a_t = actions[i] # Action taken from s_t (0 or 1)

            s_t_tensor = start_to_tensor(s_t).unsqueeze(0) # Convert partial state to tensor for policy input
            pf_logits = forward_policy(s_t_tensor)
            log_pf_s_t_to_a_t = F.log_softmax(pf_logits, dim=-1)[0, a_t] # Probability of taking action a_t from s_t
            sum_log_pf += log_pf_s_t_to_a_t

        # For this problem, we assume P_B(s_t|s_{t+1}) = 1, so log P_B = 0.
        # This is because each state s_{t+1} has a unique parent s_t by removing the last bit.
        sum_log_pb = torch.tensor(0.0, device=log_z.device)

        # Trajectory Balance Loss: (log Z + sum_t log P_F(s_{t+1}|s_t)) - (log R(x) + sum_t log P_B(s_t|s_{t+1}))
        # We take the squared difference for the loss
        loss_trajectory = (log_z + sum_log_pf - log_reward - sum_log_pb).pow(2)
        losses.append(loss_trajectory)

    return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0, device=log_z.device)