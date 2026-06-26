from tqdm.auto import tqdm
from utils import *

# Ensure N, num_episodes, batch_size, target_ones, forward_policy, flow_network, optimizer, losses, rewards are defined from previous cells
def train(num_episodes,batch_size,N,forward_policy,reward_function,target_ones,optimizer,gfn_loss,flow_network):
    losses=[]
    rewards=[]
    for episode in tqdm(range(num_episodes)):
        batch_trajectories = [] # This will store the full trajectory dictionaries
        batch_rewards = []    # This will store only the rewards for metric tracking

        for _ in range(batch_size):
            trajectory = sample_trajectory(forward_policy, N, reward_function, target_ones)
            batch_trajectories.append(trajectory)
            batch_rewards.append(trajectory['reward'])

        optimizer.zero_grad()

        # Pass the list of full trajectory dictionaries to gfn_loss
        loss = gfn_loss(forward_policy, flow_network, batch_trajectories)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        avg_reward = sum(batch_rewards) / batch_size
        rewards.append(avg_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | Loss: {loss.item():.4f} | Avg Reward: {avg_reward:.4f}")
    return losses,rewards