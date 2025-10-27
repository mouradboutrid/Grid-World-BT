import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from GridWorldEnv import GridWorldEnv
from GridWorldAnimator import GridWorldAnimator
from collections import deque

# N-Network for Q-learning
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Encode the environment state as a flat vector
def encode_state(env):
    agent = np.array(env.agent_pos, dtype=np.float32)
    goals = np.array(env.get_goal_positions(), dtype=np.float32).flatten()
    obstacles = np.array(env.get_obstacle_positions(), dtype=np.float32).flatten()
    return np.concatenate([agent, goals, obstacles])

def train_dqn(env, episodes=500, gamma=0.95, epsilon=0.2, lr=0.001):
    state_dim = len(encode_state(env))
    n_actions = len(env.action_space)
    q_net = QNetwork(state_dim, n_actions)
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    memory = deque(maxlen=2000)
    batch_size = 64

    rewards_log = []
    losses_log = []
    avg_q_log = []

    for ep in range(episodes):
        env.reset()
        state = encode_state(env)
        total_reward = 0
        total_q = 0
        steps = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action_idx = random.randint(0, n_actions - 1)
            else:
                with torch.no_grad():
                    q_values = q_net(torch.tensor(state, dtype=torch.float32))
                    action_idx = torch.argmax(q_values).item()

            action = env.action_space[action_idx]
            next_state, reward, done, _ = env.step(action)
            next_state = encode_state(env)

            # Save experience
            memory.append((state, action_idx, reward, next_state, done))
            total_reward += reward
            state = next_state
            steps += 1

            # Track Q-values
            with torch.no_grad():
                q_values_step = q_net(torch.tensor(state, dtype=torch.float32))
                total_q += q_values_step.mean().item()

            # Update network if enough experiences
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                s_batch, a_batch, r_batch, ns_batch, d_batch = zip(*batch)

                s_batch = torch.tensor(s_batch, dtype=torch.float32)
                a_batch = torch.tensor(a_batch, dtype=torch.long)
                r_batch = torch.tensor(r_batch, dtype=torch.float32)
                ns_batch = torch.tensor(ns_batch, dtype=torch.float32)
                d_batch = torch.tensor(d_batch, dtype=torch.float32)

                # Current Q-values
                q_values = q_net(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze()
                # Target Q-values
                with torch.no_grad():
                    next_q = q_net(ns_batch).max(1)[0]
                    target = r_batch + gamma * next_q * (1 - d_batch)

                # Compute loss and backprop
                loss = loss_fn(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses_log.append(loss.item())  # track loss

        rewards_log.append(total_reward)
        avg_q_log.append(total_q / max(steps, 1))  # average Q per episode

        # occasional logging
        if (ep + 1) % 50 == 0:
            avg_r = np.mean(rewards_log[-50:])
            avg_l = np.mean(losses_log[-50:]) if losses_log else 0
            avg_q = np.mean(avg_q_log[-50:])
            print(f"Episode {ep+1}/{episodes} | Avg Reward: {avg_r:.2f} | Avg Loss: {avg_l:.4f} | Avg Q: {avg_q:.4f}")

    return q_net, rewards_log, losses_log, avg_q_log

# Build a Q-table from the trained network
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

# Create environment
env = GridWorldEnv(
    size=6,
    goals=[{'pos_init': (6, 6), 'moving': False}],
    obstacles=[{'pos_init': (3, 3), 'moving': False}]
)

# Train DQN
q_net, rewards_log, losses_log, avg_q_log = train_dqn(env, episodes=300)

# Plot training progress
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

# Convert trained network to Q-table and animate
q_table = build_q_table_from_network(env, q_net)
animator = GridWorldAnimator(env, q_table, max_steps=30)
animator.animate()
