import numpy as np
import random
from GridWorldEnvV0 import GridWorldEnv
import matplotlib.pyplot as plt
from collections import defaultdict

class QLearningAgent:
    def __init__(self, env, episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = defaultdict(lambda: {a: 0.0 for a in self.env.action_space})
        self.policy = {}

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.env.action_space)
        else:
            q_values = self.Q[state]
            return max(q_values, key=q_values.get)

    def train(self):
        for ep in range(self.episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)

                best_next = max(self.Q[next_state].values()) if not done else 0
                td_target = reward + self.gamma * best_next
                td_error = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_error

                state = next_state

        # Extract policy
        for state in self.Q:
            self.policy[state] = max(self.Q[state], key=self.Q[state].get)

    def print_policy(self):
        print("Q-Learning Policy:")
        for x in range(self.env.size):
            row = []
            for y in range(self.env.size):
                state = (x, y)
                if state == self.env.terminal_state:
                    row.append("T")
                elif state in self.policy:
                    row.append(self.policy[state][0].upper())
                else:
                    row.append("?")
            print(" ".join(row))

    def plot_values(self):
        V = np.zeros((self.env.size, self.env.size))
        for (x, y), actions in self.Q.items():
            V[x, y] = max(actions.values())

        plt.imshow(V, cmap='cool', origin='upper')
        plt.colorbar()
        plt.title("State Values from Q-Learning")
        plt.gca().invert_yaxis()
        plt.show()


env = GridWorldEnv(size=5)
agent = QLearningAgent(env)
agent.train()
agent.print_policy()
agent.plot_values()
