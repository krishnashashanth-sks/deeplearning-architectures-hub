import torch
import numpy as np

def evaluate(num_evaluation_episodes,env_eval,evaluation_epsilon,main_q_network,num_evaluation_episodes_dqn):
 dqn_rewards_evaluation_episodes=[]
 for episode in range(num_evaluation_episodes):
  observation,info=env_eval.reset()
  current_state=torch.tensor(observation,dtype=torch.float32).unsqueeze(0)
  terminated=False
  truncated=False
  total_reward_episode=0
  while not terminated and not truncated:
    if np.random.uniform(0,1)>evaluation_epsilon:
      with  torch.no_grad():
        q_values=main_q_network(current_state)
        action=torch.argmax(q_values).item()
    else:
      action=env_eval.action_space.sample()
    new_observation,reward,terminated,truncated,info=env_eval.step(action)
    new_state=torch.tensor(new_observation,dtype=torch.float32).unsqueeze(0)
    total_reward_episode+=1
    current_state = new_state

  dqn_rewards_evaluation_episodes.append(total_reward_episode)
  print(f"Evaluation Episode {episode + 1}: Total Reward = {total_reward_episode}")

# 6. Calculate and print the average reward over all evaluation episodes
  average_dqn_evaluation_reward = np.mean(dqn_rewards_evaluation_episodes)
 print(f"\nDQN Evaluation complete. Average reward over {num_evaluation_episodes_dqn} episodes: {average_dqn_evaluation_reward:.2f}")

# 7. Close the environment
 env_eval.close()
 return dqn_rewards_evaluation_episodes