import gymnasium as gym
from stable_baselines3 import PPO, DQN

# Create the environment
env = gym.make("FrozenLake-v1", is_slippery=True)

# Create default PPO and DQN models
ppo_model = PPO("MlpPolicy", env, verbose=0)
dqn_model = DQN("MlpPolicy", env, verbose=0)

# Extract hyperparameters for PPO
ppo_hyperparams = {
    "learning_rate": ppo_model.learning_rate,
    "n_steps": ppo_model.n_steps,
    "batch_size": ppo_model.batch_size,
    "gamma": ppo_model.gamma,
    "ent_coef": ppo_model.ent_coef,
    "clip_range": ppo_model.clip_range,
    "gae_lambda": ppo_model.gae_lambda,
    "max_grad_norm": ppo_model.max_grad_norm,
}

# Extract hyperparameters for DQN
dqn_hyperparams = {
    "learning_rate": dqn_model.learning_rate,
    "buffer_size": dqn_model.buffer_size,
    "learning_starts": dqn_model.learning_starts,
    "batch_size": dqn_model.batch_size,
    "tau": dqn_model.tau,
    "gamma": dqn_model.gamma,
    "train_freq": dqn_model.train_freq,
    "target_update_interval": dqn_model.target_update_interval,
}

print("PPO hyperparameters for FrozenLake-v1:")
for k, v in ppo_hyperparams.items():
    print(f"{k}: {v}")

print("\nDQN hyperparameters for FrozenLake-v1:")
for k, v in dqn_hyperparams.items():
    print(f"{k}: {v}")
