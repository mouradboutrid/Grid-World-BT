
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from gymnasium import spaces

from GridWorldEnv import GridWorldEnv
from GridWorldAnimator import GridWorldAnimator
from SB3_env import make_sb3_env


#  Q-Network
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

#  DQN Training Loop (Simplified, Gym-Compatible)
def train_dqn(env, episodes=300, gamma=0.95, epsilon=0.2, lr=0.001):
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    q_net = QNetwork(state_dim, n_actions)
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    memory = deque(maxlen=2000)
    batch_size = 64

    rewards_log, losses_log, avg_q_log = [], [], []

    for ep in range(episodes):
        obs, _ = env.reset()
        state = obs / (env.size - 1)
        total_reward, total_q, steps = 0, 0, 0
        done = False

        while not done:
            # Epsilon-greedy policy
            if random.random() < epsilon:
                action_idx = random.randint(0, n_actions - 1)
            else:
                with torch.no_grad():
                    q_values = q_net(torch.tensor(state, dtype=torch.float32))
                    action_idx = torch.argmax(q_values).item()

            next_obs, reward, terminated, truncated, _ = env.step(action_idx)
            done = terminated or truncated
            next_state = next_obs / (env.size - 1)
            memory.append((state, action_idx, reward, next_state, done))
            total_reward += reward
            state = next_state
            steps += 1

            with torch.no_grad():
                q_values_step = q_net(torch.tensor(state, dtype=torch.float32))
                total_q += q_values_step.mean().item()

            # Train if enough memory
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                s_batch, a_batch, r_batch, ns_batch, d_batch = zip(*batch)

                s_batch = torch.tensor(s_batch, dtype=torch.float32)
                a_batch = torch.tensor(a_batch, dtype=torch.long)
                r_batch = torch.tensor(r_batch, dtype=torch.float32)
                ns_batch = torch.tensor(ns_batch, dtype=torch.float32)
                d_batch = torch.tensor(d_batch, dtype=torch.float32)

                q_values = q_net(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    next_q = q_net(ns_batch).max(1)[0]
                    target = r_batch + gamma * next_q * (1 - d_batch)

                loss = loss_fn(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                optimizer.step()
                losses_log.append(loss.item())

        rewards_log.append(total_reward)
        avg_q_log.append(total_q / max(steps, 1))

        if (ep + 1) % 50 == 0:
            avg_r = np.mean(rewards_log[-50:])
            avg_l = np.mean(losses_log[-50:]) if losses_log else 0
            avg_q = np.mean(avg_q_log[-50:])
            print(f"Episode {ep+1}/{episodes} | "
                  f"Avg Reward: {avg_r:.2f} | Avg Loss: {avg_l:.4f} | Avg Q: {avg_q:.4f}")

    return q_net, rewards_log, losses_log, avg_q_log


#  Build Q-Table for Visualization
def build_q_table_from_network(env, q_net):
    q_table = {}
    for x in range(env.size):
        for y in range(env.size):
            obs = np.array([x, y], dtype=np.float32) / (env.size - 1)
            with torch.no_grad():
                q_values = q_net(torch.tensor(obs, dtype=torch.float32))
            q_table[(x, y)] = q_values.numpy()
    return q_table

#  Main Execution
base_env = GridWorldEnv(
    size=6,
    goals=[{'pos_init': (5, 5), 'moving': False}],
    obstacles=[{'pos_init': (3, 3), 'moving': False}]
)
env = make_sb3_env(base_env)
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
q_table = build_q_table_from_network(base_env, q_net)
animator = GridWorldAnimator(base_env, q_table, max_steps=30)
animator.animate()
