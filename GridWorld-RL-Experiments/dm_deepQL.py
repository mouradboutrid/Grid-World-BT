import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from GridWorldEnv import GridWorldEnv
from GridWorldAnimator import GridWorldAnimator

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# State encoding & normalization
def encode_state(env):
    agent = np.array(env.agent_pos, dtype=np.float32) / env.size
    goals = np.array(env.get_goal_positions(), dtype=np.float32).flatten() / env.size
    obstacles = np.array(env.get_obstacle_positions(), dtype=np.float32).flatten() / env.size
    return np.concatenate([agent, goals, obstacles])

# DQN training
def train_dqn(env, episodes=300, gamma=0.99, lr=0.001,
              epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
              batch_size=64, memory_size=5000, target_update=10):
    
    state_dim = len(encode_state(env))
    n_actions = len(env.action_space)
    
    q_net = QNetwork(state_dim, n_actions)
    target_net = QNetwork(state_dim, n_actions)
    target_net.load_state_dict(q_net.state_dict())  # sync networks
    target_net.eval()
    
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    memory = deque(maxlen=memory_size)
    
    epsilon = epsilon_start
    rewards_log = []
    losses_log = []
    avg_q_log = []
    
    for ep in range(episodes):
        state = encode_state(env)
        env.reset()
        total_reward = 0
        total_q = 0
        steps = 0
        done = False
        
        while not done:
            # Epsilon-greedy
            if random.random() < epsilon:
                action_idx = random.randint(0, n_actions - 1)
            else:
                with torch.no_grad():
                    q_values = q_net(torch.tensor(state, dtype=torch.float32))
                    action_idx = torch.argmax(q_values).item()
            
            action = env.action_space[action_idx]
            next_state, reward, done, _ = env.step(action)
            next_state = encode_state(env)
            
            # Clip reward
            reward = np.clip(reward, -1, 1)
            
            memory.append((state, action_idx, reward, next_state, done))
            state = next_state
            total_reward += reward
            steps += 1
            
            # Track Q-values
            with torch.no_grad():
                q_values_step = q_net(torch.tensor(state, dtype=torch.float32))
                total_q += q_values_step.mean().item()
            
            # Update network
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                s_batch, a_batch, r_batch, ns_batch, d_batch = zip(*batch)
                
                s_batch = torch.tensor(s_batch, dtype=torch.float32)
                a_batch = torch.tensor(a_batch, dtype=torch.long)
                r_batch = torch.tensor(r_batch, dtype=torch.float32)
                ns_batch = torch.tensor(ns_batch, dtype=torch.float32)
                d_batch = torch.tensor(d_batch, dtype=torch.float32)
                
                q_values_batch = q_net(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze()
                
                with torch.no_grad():
                    next_q = target_net(ns_batch).max(1)[0]
                    target = r_batch + gamma * next_q * (1 - d_batch)
                
                loss = loss_fn(q_values_batch, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses_log.append(loss.item())
        
        # Average Q-value for this episode
        avg_q_log.append(total_q / max(steps, 1))
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_log.append(total_reward)
        
        # Update target network
        if (ep + 1) % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())
        
        # Logging
        if (ep + 1) % 50 == 0:
            avg_r = np.mean(rewards_log[-50:])
            avg_l = np.mean(losses_log[-50:]) if losses_log else 0
            avg_q = np.mean(avg_q_log[-50:])
            print(f"Episode {ep+1}/{episodes} | Avg Reward: {avg_r:.2f} | Avg Loss: {avg_l:.4f} | Avg Q: {avg_q:.4f} | Epsilon: {epsilon:.2f}")
    
    return q_net, rewards_log, losses_log, avg_q_log

# Q-table build for animation
def build_q_table_from_network(env, q_net):
    q_table = {}
    for x in range(env.size):
        for y in range(env.size):
            env.agent_pos = (x, y)
            state = encode_state(env)
            with torch.no_grad():
                q_values = q_net(torch.tensor(state, dtype=torch.float32))
            q_table[(x, y)] = q_values.numpy()
    return q_table

# Training & animation
env = GridWorldEnv(
    size=6,
    goals=[{'pos_init': (5, 5), 'moving': False}],
    obstacles=[{'pos_init': (2, 2), 'moving': True}]
)

q_net, rewards_log, losses_log, avg_q_log = train_dqn(env, episodes=300)

# Plot progress
plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
plt.plot(rewards_log, label='Total Reward')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Reward Progress")
plt.legend()

plt.subplot(1,3,2)
plt.plot(losses_log, label='Loss', color='red')
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("DQN Loss Progress")
plt.legend()

plt.subplot(1,3,3)
plt.plot(avg_q_log, label='Avg Q-value', color='green')
plt.xlabel("Episode")
plt.ylabel("Q-value")
plt.title("Average Q-value per Episode")
plt.legend()

plt.tight_layout()
plt.show()

# Animate
q_table = build_q_table_from_network(env, q_net)
animator = GridWorldAnimator(env, q_table, max_steps=30)
animator.animate()
