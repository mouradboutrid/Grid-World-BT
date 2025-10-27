import random
import numpy as np
import matplotlib.pyplot as plt
from GridWorldEnv import GridWorldEnv
from GridWorldAnimator import GridWorldAnimator
from collections import defaultdict

# Setup
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 1000
max_steps = 50

# Create environment
env = GridWorldEnv(
    size=6,
    goals=[{'pos_init': (5, 5), 'moving': False}],
    obstacles=[{'pos_init': (2, 2), 'moving': False}]
)

# Q-table
Q = defaultdict(lambda: np.zeros(len(env.action_space)))

def choose_action(state):
    if random.random() < epsilon:
        return random.choice(range(len(env.action_space)))
    return np.argmax(Q[state])

# Train Q-learning
for ep in range(episodes):
    state = env.reset()
    done = False
    for _ in range(max_steps):
        state_key = tuple(state)
        action_idx = choose_action(state_key)
        action = env.action_space[action_idx]

        next_state, reward, done, _ = env.step(action)
        next_key = tuple(next_state)

        best_next = np.max(Q[next_key])
        Q[state_key][action_idx] += alpha * (reward + gamma * best_next - Q[state_key][action_idx])

        state = next_state
        if done:
            break

print("Training complete!")



animator = GridWorldAnimator(env, Q, max_steps=50)
animator.animate()
