import numpy as np
from stable_baselines3 import PPO
from GridWorldEnv import GridWorldEnv
from SB3_env import make_sb3_env
from GridWorldAnimator import GridWorldAnimator

# Create and wrap environment
env_orig = GridWorldEnv(
    size=6,
    goals=[{'pos_init': (4,4), 'moving': True}],       # moving goal
    obstacles=[{'pos_init': (2,2), 'moving': True}]    # moving obstacle
)
env = make_sb3_env(env_orig)

# Use FrozenLake PPO hyperparameters
ppo_hyperparams = {
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "ent_coef": 0.0,
    "clip_range": 0.2,
    "gae_lambda": 0.95,
    "max_grad_norm": 0.5
}

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/gridworld_tensorboard/",
    learning_rate=ppo_hyperparams["learning_rate"],
    n_steps=ppo_hyperparams["n_steps"],
    batch_size=ppo_hyperparams["batch_size"],
    gamma=ppo_hyperparams["gamma"],
    ent_coef=ppo_hyperparams["ent_coef"],
    clip_range=ppo_hyperparams["clip_range"],
    gae_lambda=ppo_hyperparams["gae_lambda"],
    max_grad_norm=ppo_hyperparams["max_grad_norm"]
)

# Train PPO
model.learn(total_timesteps=50000)
model.save("ppo_gridworld")
env.close()

# Animate trained agent
env_anim = GridWorldEnv(
    size=6,
    goals=[{'pos_init': (4,4), 'moving': True}],
    obstacles=[{'pos_init': (2,2), 'moving': True}]
)

# Policy wrapper for animator
def policy_fn(state):
    # PPO expects 2D array for predict
    action_index, _ = model.predict(np.array(state).reshape(1,-1), deterministic=True)
    return env_anim.action_space[action_index[0]]  # convert index to string

animator = GridWorldAnimator(env_anim, policy=policy_fn, max_steps=50)
animator.animate()
