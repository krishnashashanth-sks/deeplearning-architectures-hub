import torch
import numpy as np
import numpy as np
import torch

def train(num_episodes, env, main_q_network, replay_buffer, batch_size, 
          target_q_network, gamma, criterion, optimizer, 
          epsilon, epsilon_min, epsilon_decay, target_update_frequency):
    
    dqn_rewards_per_episode = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        
        # FIX: Consistently add batch dimension (1, state_dim) for the network
        current_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        terminated = False
        truncated = False  # FIX: Removed trailing comma typo
        total_reward_episode = 0
        episode_steps = 0
        
        while not terminated and not truncated:
            # Action selection (Epsilon-Greedy)
            if np.random.uniform(0, 1) > epsilon:
                with torch.no_grad():
                    q_values = main_q_network(current_state)
                    action = torch.argmax(q_values).item()
            else:
                action = env.action_space.sample()
            
            # Environment step
            next_state_np, reward, terminated, truncated, info = env.step(action)
            
            # FIX: Create next_state tensor with batch dimension cleanly
            next_state = torch.tensor(next_state_np, dtype=torch.float32).unsqueeze(0)
            
            # Store experience in replay buffer (saving numpy arrays is standard practice)
            replay_buffer.add(state, action, reward, next_state_np, terminated or truncated)
            
            total_reward_episode += reward
            current_state = next_state
            state = next_state_np
            episode_steps += 1
            
            # Optimization / Gradient Descent Step
            if len(replay_buffer) > batch_size:
                experiences = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = experiences
                
                # NOTE: If your replay buffer returns numpy arrays, convert them to torch tensors here.
                # e.g., states = torch.tensor(states, dtype=torch.float32)
                
                predicted_q_values = main_q_network(states).gather(1, actions)
                
                with torch.no_grad():
                    max_next_q_values = target_q_network(next_states).max(1).unsqueeze(1)
                    target_q_values = rewards + gamma * max_next_q_values * (1 - dones)
                
                loss = criterion(predicted_q_values, target_q_values)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Track total episode rewards
        dqn_rewards_per_episode.append(total_reward_episode)

        # FIX: Moved Epsilon Decay here so it decays per episode, not per step
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Periodically update the target network
        if (episode + 1) % target_update_frequency == 0:
            target_q_network.load_state_dict(main_q_network.state_dict())

        # Print training progress
        if (episode + 1) % 100 == 0:
            average_reward_last_100 = np.mean(dqn_rewards_per_episode[-100:])
            print(f"Episode {episode + 1}: Average reward over last 100 episodes: {average_reward_last_100:.2f}, Epsilon: {epsilon:.4f}")

    env.close()
    print("DQN Training complete.")
    return dqn_rewards_per_episode