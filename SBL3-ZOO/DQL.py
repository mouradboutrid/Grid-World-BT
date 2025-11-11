from stable_baselines3 import DQN
from GridWorldEnv import GridWorldEnv
from SB3_env import make_sb3_env
from GridWorldAnimator import GridWorldAnimator
import numpy as np 

env_orig = GridWorldEnv(
    size=6,
    goals=[{'pos_init': (4,4), 'moving': True}],      # Moving goal
    obstacles=[{'pos_init': (2,2), 'moving': True}]   # Moving obstacle
)
env = make_sb3_env(env_orig)

# Create DQN model
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    tensorboard_log="./logs/gridworld_tensorboard_dqn/"
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

# Define policy wrapper for animator
def policy_fn(state):
    # DQN expects 2D array
    action_index, _ = model.predict(np.array(state).reshape(1,-1), deterministic=True)
    return env_anim.action_space[action_index[0]]  # convert index to string action

# Run animation
animator = GridWorldAnimator(env_anim, policy=policy_fn, max_steps=50)
animator.animate()
