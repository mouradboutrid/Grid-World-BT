import numpy as np
import gymnasium as gym
from gymnasium import spaces


def make_sb3_env(original_env):
    """
    Wraps the original GridWorldEnv to make it compatible with Stable Baselines3.
    """

    class SB3GridWrapper(gym.Env):
        metadata = {"render_modes": ["human"]}

        def __init__(self, env):
            super(SB3GridWrapper, self).__init__()
            self.env = env
            self.size = env.size

            self.action_space = spaces.Discrete(len(env.action_space))
            self.observation_space = spaces.Box(
                low=0, high=self.size - 1, shape=(2,), dtype=np.int32
            )

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                np.random.seed(seed)
            obs = np.array(self.env.reset(), dtype=np.int32)
            info = {}
            return obs, info

        def step(self, action):
            # Convert numeric action to string for original environment
            action_str = self.env.action_space[action]
            new_pos, reward, done, _ = self.env.step(action_str)
            obs = np.array(new_pos, dtype=np.int32)
            terminated = done
            truncated = False
            info = {}
            return obs, reward, terminated, truncated, info

        def render(self):
            self.env.render()

        def close(self):
            pass

    # Return a wrapped SB3-compatible environment
    return SB3GridWrapper(original_env)
