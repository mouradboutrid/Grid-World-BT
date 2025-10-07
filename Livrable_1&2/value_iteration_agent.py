import numpy as np
from GridWorldEnvV0 import GridWorldEnv
import matplotlib.pyplot as plt

class ValueIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-4):
        self.env = env
        self.gamma = gamma      # Discount factor
        self.theta = theta      # Convergence threshold
        self.V = np.zeros((env.size, env.size))
        self.policy = np.full((env.size, env.size), '', dtype=object)

    def one_step_lookahead(self, state):
        x, y = state
        actions = self.env.action_space
        values = {}

        for action in actions:
            next_x, next_y = x, y

            if action == 'up':
                next_x = max(0, x - 1)
            elif action == 'down':
                next_x = min(self.env.size - 1, x + 1)
            elif action == 'left':
                next_y = max(0, y - 1)
            elif action == 'right':
                next_y = min(self.env.size - 1, y + 1)

            next_state = (next_x, next_y)
            reward = 1.0 if next_state == self.env.terminal_state else 0.0

            values[action] = reward + self.gamma * self.V[next_x, next_y]

        return values

    def value_iteration(self):
        iteration = 0
        while True:
            delta = 0
            for x in range(self.env.size):
                for y in range(self.env.size):
                    state = (x, y)
                    if state == self.env.terminal_state:
                        continue  # Skip terminal

                    action_values = self.one_step_lookahead(state)
                    best_action_value = max(action_values.values())
                    delta = max(delta, abs(self.V[x, y] - best_action_value))
                    self.V[x, y] = best_action_value

            iteration += 1
            if delta < self.theta:
                break

        # Extract policy
        for x in range(self.env.size):
            for y in range(self.env.size):
                state = (x, y)
                if state == self.env.terminal_state:
                    self.policy[x, y] = 'T'  # Terminal
                    continue
                action_values = self.one_step_lookahead(state)
                best_action = max(action_values, key=action_values.get)
                self.policy[x, y] = best_action

    def print_policy(self):
        print("Optimal Policy:")
        for row in self.policy:
            print(' '.join(row))

    def plot_values(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.env.size)
        ax.set_ylim(0, self.env.size)
        ax.set_xticks(np.arange(0, self.env.size + 1))
        ax.set_yticks(np.arange(0, self.env.size + 1))
        ax.grid(True)

        for x in range(self.env.size):
            for y in range(self.env.size):
                value = self.V[x, y]
                rect = plt.Rectangle((y, self.env.size - 1 - x), 1, 1, fill=False)
                ax.add_patch(rect)

                # Color terminal state
                if (x, y) == self.env.terminal_state:
                    ax.add_patch(plt.Rectangle((y, self.env.size - 1 - x), 1, 1, color='green', alpha=0.5))

                # Add value text
                ax.text(y + 0.5, self.env.size - 1 - x + 0.5,
                        f"{value:.2f}", ha='center', va='center', fontsize=10)

        ax.set_title("State Values (V)")
        ax.set_aspect('equal')
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.show()

    def plot_policy(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.env.size)
        ax.set_ylim(0, self.env.size)
        ax.set_xticks(np.arange(0, self.env.size + 1))
        ax.set_yticks(np.arange(0, self.env.size + 1))
        ax.grid(True)

        for x in range(self.env.size):
            for y in range(self.env.size):
                action = self.policy[x, y]
                if action == 'T':
                    ax.add_patch(plt.Rectangle((y, self.env.size - 1 - x), 1, 1, color='green'))
                    ax.text(y + 0.5, self.env.size - 1 - x + 0.5, 'T', ha='center', va='center', color='white')
                else:
                    # Display first letter of action (U, D, L, R)
                    ax.text(y + 0.5, self.env.size - 1 - x + 0.5,
                            action[0].upper(), ha='center', va='center', fontsize=12)

        plt.title("Optimal Policy (First Letter of Action)")
        ax.set_aspect('equal')
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.show()



env = GridWorldEnv(size=5)
agent = ValueIterationAgent(env)
agent.value_iteration()
agent.print_policy()
agent.plot_values()
agent.plot_policy()
