import numpy as np
from stable_baselines3 import A2C
from GridWorldEnv import GridWorldEnv
from SB3_env import make_sb3_env
from GridWorldAnimator import GridWorldAnimator

# Create and wrap environment
env_orig = GridWorldEnv(
    size=6,
    goals=[{'pos_init': (4, 4), 'moving': True}],       # Moving goal
    obstacles=[{'pos_init': (2, 2), 'moving': True}]    # Moving obstacle
)
env = make_sb3_env(env_orig)

# A2C hyperparameters (defaults similar to SB3 frozenlake)
a2c_hyperparams = {
    "learning_rate": 0.0007,
    "n_steps": 5,
    "gamma": 0.99,
    "gae_lambda": 1.0,
    "ent_coef": 0.0,
    "vf_coef": 0.25,
    "max_grad_norm": 0.5
}

# Create A2C model
model = A2C(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=a2c_hyperparams["learning_rate"],
    n_steps=a2c_hyperparams["n_steps"],
    gamma=a2c_hyperparams["gamma"],
    gae_lambda=a2c_hyperparams["gae_lambda"],
    ent_coef=a2c_hyperparams["ent_coef"],
    vf_coef=a2c_hyperparams["vf_coef"],
    max_grad_norm=a2c_hyperparams["max_grad_norm"],
    tensorboard_log="./logs/gridworld_tensorboard_a2c/"
)

# Train A2C
model.learn(total_timesteps=50000)
model.save("a2c_gridworld")
env.close()

# Animate trained agent

env_anim = GridWorldEnv(
    size=6,
    goals=[{'pos_init': (4, 4), 'moving': True}],
    obstacles=[{'pos_init': (2, 2), 'moving': True}]
)

# Policy wrapper for animator
def policy_fn(state):
    # A2C expects 2D array for predict
    action_index, _ = model.predict(np.array(state).reshape(1, -1), deterministic=True)
    return env_anim.action_space[action_index[0]]  # Convert index to string

animator = GridWorldAnimator(env_anim, policy=policy_fn, max_steps=50)
animator.animate()
