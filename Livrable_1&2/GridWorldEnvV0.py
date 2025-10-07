import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import numpy as np

class GridWorldEnv:
    def __init__(self, size=5):
        self.size = size
        self.action_space = ['up', 'down', 'left', 'right']
        self.terminal_state = (size - 1, size - 1)
        self.reset()

    def reset(self):
        self.agent_pos = (0, 0)
        self.done = False
        return self.agent_pos

    def step(self, action):
        if self.done:
            raise Exception("Episode is done. Call reset() to start a new episode.")
        
        x, y = self.agent_pos

        if action == 'up':
            x = max(0, x - 1)
        elif action == 'down':
            x = min(self.size - 1, x + 1)
        elif action == 'left':
            y = max(0, y - 1)
        elif action == 'right':
            y = min(self.size - 1, y + 1)

        self.agent_pos = (x, y)
        
        if self.agent_pos == self.terminal_state:
            reward = 50.0
            self.done = True
        else:
            reward = -0.5

        return self.agent_pos, reward, self.done, {}

    def render(self):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_xticks(range(self.size + 1))
        ax.set_yticks(range(self.size + 1))
        ax.grid(True)

        # Draw the grid world
        for i in range(self.size):
            for j in range(self.size):
                rect = Rectangle((j, self.size - 1 - i), 1, 1, fill=False, edgecolor='black')
                ax.add_patch(rect)

        # Draw terminal state in black
        tx, ty = self.terminal_state
        terminal_rect = Rectangle((ty, self.size - 1 - tx), 1, 1, color='black')
        ax.add_patch(terminal_rect)

        # Draw agent in blue
        x, y = self.agent_pos
        agent_rect = Rectangle((y, self.size - 1 - x), 1, 1, color='blue')
        ax.add_patch(agent_rect)

        ax.set_title(f"Agent at {self.agent_pos}")
        ax.set_aspect('equal')
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.show()



env = GridWorldEnv(size=5)
state = env.reset()
done = False
step_count = 0

while not done:
    action = random.choice(env.action_space)
    state, reward, done, _ = env.step(action)
    print(f"Step {step_count}: Action={action}, State={state}, Reward={reward}")
    env.render()
    step_count += 1
