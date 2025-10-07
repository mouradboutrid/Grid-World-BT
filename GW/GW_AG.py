import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches


class GridWorldAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}

    def get_state_key(self, pos):
        return tuple(pos)

    def choose_action(self, state_key):
        if random.uniform(0, 1) < self.epsilon or state_key not in self.q_table:
            return random.choice(self.env.action_space)
        else:
            return self.env.action_space[np.argmax(self.q_table[state_key])]

    def update_q_table(self, state_key, action, reward, next_state_key, done):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.env.action_space))
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(len(self.env.action_space))

        action_index = self.env.action_space.index(action)
        next_max = np.max(self.q_table[next_state_key])
        target = reward if done else reward + self.gamma * next_max
        self.q_table[state_key][action_index] += self.alpha * (target - self.q_table[state_key][action_index])

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class GridWorldTrainer:
    def __init__(self, env, episodes=2000, max_steps=100,
                 alpha=0.1, gamma=0.95,
                 epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 moving_avg_window=50):
        self.env = env
        self.grid_size = env.size
        self.episodes = episodes
        self.max_steps = max_steps
        self.moving_avg_window = moving_avg_window

        self.agent = GridWorldAgent(
            env=self.env,
            alpha=alpha,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min
        )

        self.rewards_all_episodes = []
        self.steps_all_episodes = []

    def train(self):
        for episode in range(1, self.episodes + 1):
            state = self.env.reset()
            state_key = self.agent.get_state_key(state)
            total_reward = 0

            for step in range(self.max_steps):
                action = self.agent.choose_action(state_key)
                next_state, reward, done, _ = self.env.step(action)
                next_state_key = self.agent.get_state_key(next_state)

                self.agent.update_q_table(state_key, action, reward, next_state_key, done)

                state_key = next_state_key
                total_reward += reward

                if done:
                    break

            self.rewards_all_episodes.append(total_reward)
            self.steps_all_episodes.append(step + 1)
            self.agent.decay_epsilon()

        moving_avg_rewards = np.convolve(
            self.rewards_all_episodes,
            np.ones(self.moving_avg_window) / self.moving_avg_window,
            mode='valid'
        )

        return {
            'env': self.env,
            'q_table': self.agent.q_table,
            'rewards': self.rewards_all_episodes,
            'moving_avg_rewards': moving_avg_rewards,
            'steps': self.steps_all_episodes,
            'grid_size': self.grid_size
        }


class GridWorldAnimator:
    def __init__(self, env, q_table, max_steps=30):
        self.env = env
        self.q_table = q_table
        self.max_steps = max_steps

    def animate(self):
        frames = []
        state = self.env.reset()
        state_key = tuple(state)
        done = False

        for _ in range(self.max_steps):
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

            if state_key in self.q_table:
                action = self.env.action_space[np.argmax(self.q_table[state_key])]
            else:
                action = random.choice(self.env.action_space)

            state, _, done, _ = self.env.step(action)
            state_key = tuple(state)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, self.env.size)
        ax.set_ylim(0, self.env.size)
        ax.set_xticks(np.arange(0, self.env.size + 1, 1))
        ax.set_yticks(np.arange(0, self.env.size + 1, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)
        ax.set_title(f"GridWorld Agent Navigation (Size {self.env.size})")

        background_cells = [[
            ax.add_patch(patches.Rectangle((x, self.env.size - y - 1), 1, 1, edgecolor='gray', facecolor='white'))
            for y in range(self.env.size)]
            for x in range(self.env.size)
        ]

        cell_patches = [[
            ax.add_patch(patches.Rectangle((x, self.env.size - y - 1), 1, 1, edgecolor='gray', facecolor='white'))
            for y in range(self.env.size)]
            for x in range(self.env.size)
        ]

        def update_grid(frame):
            grid = frame
            for x in range(self.env.size):
                for y in range(self.env.size):
                    value = grid[x][y]
                    patch = cell_patches[x][y]
                    if value == 'A':
                        patch.set_facecolor('green')
                    elif value == 'G':
                        patch.set_facecolor('red')
                    elif value == 'X':
                        patch.set_facecolor('black')
                    else:
                        patch.set_facecolor('white')
            return sum(cell_patches, [])

        ani = animation.FuncAnimation(fig, update_grid, frames=frames, interval=500, blit=True)
        plt.show()
