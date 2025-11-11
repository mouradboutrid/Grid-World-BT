from stable_baselines3 import PPO
from GridWorldEnv import GridWorldEnv
from SB3_env import make_sb3_env
from GridWorldAnimator import GridWorldAnimator
import numpy as np 
# Create and wrap environment
env_orig = GridWorldEnv(
    size=6,
    goals=[{'pos_init': (4,4)}],
    obstacles=[{'pos_init': (2,2)}]
)
env = make_sb3_env(env_orig)

# Create PPO model

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/gridworld_tensorboard/"
)

# Train PPO
model.learn(total_timesteps=50000)
model.save("ppo_gridworld")
env.close()

# Animate trained agent
# Reload environment for animation
env_anim = GridWorldEnv(
    size=5,
    goals=[{'pos_init': (4,4)}],
    obstacles=[{'pos_init': (2,2)}]
)

# Define policy wrapper for animator
def policy_fn(state):
    # PPO expects 2D array for predict
    action_index, _ = model.predict(np.array(state).reshape(1,-1), deterministic=True)
    return env_anim.action_space[action_index[0]]  # convert index to string action

animator = GridWorldAnimator(env_anim, policy=policy_fn, max_steps=30)
animator.animate()
