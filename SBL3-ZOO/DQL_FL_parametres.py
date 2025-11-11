import numpy as np
from stable_baselines3 import DQN
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

# FrozenLake DQN hyperparameters
dqn_hyperparams = {
    "learning_rate": 0.0001,
    "buffer_size": 1000000,
    "learning_starts": 100,
    "batch_size": 32,
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": (4, "step"),
    "target_update_interval": 10000
}

# Create DQN model
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=dqn_hyperparams["learning_rate"],
    buffer_size=dqn_hyperparams["buffer_size"],
    learning_starts=dqn_hyperparams["learning_starts"],
    batch_size=dqn_hyperparams["batch_size"],
    tau=dqn_hyperparams["tau"],
    gamma=dqn_hyperparams["gamma"],
    train_freq=dqn_hyperparams["train_freq"],
    target_update_interval=dqn_hyperparams["target_update_interval"],
    tensorboard_log="./logs/gridworld_tensorboard/"
)

# Train DQN
model.learn(total_timesteps=50000)
model.save("dqn_gridworld")
env.close()

# Animate trained agent
env_anim = GridWorldEnv(
    size=6,
    goals=[{'pos_init': (4,4), 'moving': True}],
    obstacles=[{'pos_init': (2,2), 'moving': True}]
)

# Policy wrapper for animator
def policy_fn(state):
    # DQN expects 2D array for predict
    action_index, _ = model.predict(np.array(state).reshape(1,-1), deterministic=True)
    return env_anim.action_space[action_index[0]]  # convert index to string

animator = GridWorldAnimator(env_anim, policy=policy_fn, max_steps=50)
animator.animate()
