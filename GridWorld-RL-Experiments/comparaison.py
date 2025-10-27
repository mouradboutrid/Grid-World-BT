import matplotlib.pyplot as plt
from stable_baselines3 import DQN
import torch
import numpy as np

from SB3_env import make_sb3_env
from naiveDL_agent_SB3 import train_dqn as train_dqn_simple
from mydeepQL_SB3 import train_dqn as train_dqn_full, build_q_table_from_network
from GridWorldEnv import GridWorldEnv
from GridWorldAnimator import GridWorldAnimator

# Evaluation function for custom PyTorch DQNs
def evaluate_custom(env, q_net, episodes=20):
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        state = obs / (env.size - 1)  # normalize
        done = False
        ep_reward = 0
        while not done:
            with torch.no_grad():
                action = torch.argmax(q_net(torch.tensor(state, dtype=torch.float32))).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = obs / (env.size - 1)
            ep_reward += reward
        rewards.append(ep_reward)
    return np.mean(rewards), rewards

# Evaluation function for SB3 DQN
def evaluate_sb3(env, model, episodes=20):
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        rewards.append(ep_reward)
    return np.mean(rewards), rewards

# Setup environment
base_env = GridWorldEnv(
    size=6,
    goals=[{'pos_init': (5,5), 'moving': False}],
    obstacles=[{'pos_init': (3,3), 'moving': False}]
)
env = make_sb3_env(base_env)

# Train custom DQNs
q_net_full, r_log1, l_log1, q_log1 = train_dqn_full(env, episodes=300)
q_net_simple, r_log2, l_log2, q_log2 = train_dqn_simple(env, episodes=300)

# Train SB3 DQN
sb3_env = make_sb3_env(base_env)
sb3_model = DQN("MlpPolicy", sb3_env, verbose=0)
sb3_model.learn(total_timesteps=100_000)

# Evaluate all models
mean1, r1 = evaluate_custom(env, q_net_full)
mean2, r2 = evaluate_custom(env, q_net_simple)
mean_sb3, r_sb3 = evaluate_sb3(sb3_env, sb3_model)

# Plot comparison
plt.figure(figsize=(12,5))
plt.plot(r1, label="Custom DQN Full")
plt.plot(r2, label="Custom DQN Simple")
plt.plot(r_sb3, label="SB3 DQN")
plt.axhline(mean1, color='blue', linestyle='--', alpha=0.6)
plt.axhline(mean2, color='orange', linestyle='--', alpha=0.6)
plt.axhline(mean_sb3, color='green', linestyle='--', alpha=0.6)
plt.xlabel("Evaluation Episode")
plt.ylabel("Total Reward")
plt.title("Comparison: Custom DQNs vs SB3 DQN")
plt.legend()
plt.show()

# Animate custom DQNs
q_table_full = build_q_table_from_network(base_env, q_net_full)
animator_full = GridWorldAnimator(base_env, q_table_full, max_steps=30)
animator_full.animate()

q_table_simple = build_q_table_from_network(base_env, q_net_simple)
animator_simple = GridWorldAnimator(base_env, q_table_simple, max_steps=30)
animator_simple.animate()
