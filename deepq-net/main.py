import gymnasium as gym
env=gym.make('CartPole-v1')
from model import DQNetwork
from layers import ReplayBuffer
import torch.nn as nn
import torch.optim as optim
from train import train
from evaluate import evaluate

state_dim=env.observation_space.shape[0]
action_dim=env.action_space.n

# 3. Set up Main and Target Q-Networks
main_q_network = DQNetwork(state_dim, action_dim)
target_q_network = DQNetwork(state_dim, action_dim)
target_q_network.load_state_dict(main_q_network.state_dict()) # Copy weights
target_q_network.eval() # Set target network to evaluation mode (no gradients)

# Instantiate Replay Buffer
replay_buffer = ReplayBuffer(capacity=10000)

learning_rate=0.001
gamma=0.99
epsilon=1.0

epsilon_min = 0.01          # Minimum exploration rate
epsilon_decay = 0.995       # Exploration decay rate
batch_size = 64             # Size of mini-batch to sample from replay buffer
target_update_frequency = 10 # Update target network every N episodes
num_episodes = 2000         # Total number of training episodes
optimizer=optim.Adam(main_q_network.parameters(), lr=learning_rate)
criterion=nn.MSELoss()

dqn_rewards_per_episode=train(num_episodes, env, main_q_network, replay_buffer, batch_size, 
          target_q_network, gamma, criterion, optimizer, 
          epsilon, epsilon_min, epsilon_decay, target_update_frequency)

evaluation_epsilon=0.0
num_evaluation_episodes=10

dqn_rewards_evaluation_episodes=evaluate(num_evaluation_episodes,env_eval,evaluation_epsilon,main_q_network,num_evaluation_episodes_dqn)