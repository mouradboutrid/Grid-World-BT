import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

class GridWorldAnimator:
    def __init__(self, env, q_table=None, policy=None, max_steps=30):
        """
        env       : GridWorldEnv instance
        q_table   : optional dict {(x,y): q_values} for DQN
        policy    : optional function(state) -> action (for SB3 or custom policy)
        max_steps : maximum steps to animate
        """
        self.env = env
        self.q_table = q_table or {}
        self.policy = policy
        self.max_steps = max_steps

    def animate(self):
        frames = []
        state = self.env.reset()
        done = False

        for _ in range(self.max_steps):
            # Build grid
            grid = [['.' for _ in range(self.env.size)] for _ in range(self.env.size)]

            for o in self.env.get_obstacle_positions():
                x, y = o
                grid[x][y] = 'X'

            for g in self.env.get_goal_positions():
                x, y = g
                grid[x][y] = 'G'

            ax, ay = self.env.agent_pos
            grid[ax][ay] = 'A'
            frames.append(grid)

            if done:
                break

            # Choose action
            if self.policy is not None:
                action = self.policy(state)  # SB3 policy
            else:
                state_key = tuple(state)
                if state_key in self.q_table:
                    action = self.env.action_space[np.argmax(self.q_table[state_key])]
                else:
                    action = random.choice(self.env.action_space)

            # Take step
            state, _, done, _ = self.env.step(action)

        # Setup plot
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, self.env.size)
        ax.set_ylim(0, self.env.size)
        ax.set_xticks(np.arange(0, self.env.size + 1, 1))
        ax.set_yticks(np.arange(0, self.env.size + 1, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)
        ax.set_title(f"GridWorld Agent Navigation (Size {self.env.size})")

        # Background cells
        background_cells = [[
            ax.add_patch(patches.Rectangle((x, self.env.size - y - 1), 1, 1, edgecolor='gray', facecolor='white'))
            for y in range(self.env.size)]
            for x in range(self.env.size)
        ]

        # Agent/goal/obstacle cells
        cell_patches = [[
            ax.add_patch(patches.Rectangle((x, self.env.size - y - 1), 1, 1, edgecolor='gray', facecolor='white'))
            for y in range(self.env.size)]
            for x in range(self.env.size)
        ]

        # Update function
        def update_grid(frame):
            grid = frame
            for x in range(self.env.size):
                for y in range(self.env.size):
                    val = grid[x][y]
                    patch = cell_patches[x][y]
                    if val == 'A':
                        patch.set_facecolor('green')
                    elif val == 'G':
                        patch.set_facecolor('red')
                    elif val == 'X':
                        patch.set_facecolor('black')
                    else:
                        patch.set_facecolor('white')
            return sum(cell_patches, [])

        # Animate
        ani = animation.FuncAnimation(fig, update_grid, frames=frames, interval=500, blit=True)
        plt.show()