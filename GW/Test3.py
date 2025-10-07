import numpy as np
import matplotlib.pyplot as plt
from GridWorldEnv import GridWorldEnv
from GW_AG import GridWorldTrainer, GridWorldAnimator  

# === Hyperparameters ===
EPISODES = 1500
MAX_STEPS_PER_EPISODE = 100
MOVING_AVG_WINDOW = 50

# === Grid sizes to evaluate ===
grid_sizes = [4, 6, 8, 10]
results = []

for size in grid_sizes:
    print(f"\n--- Training for grid size {size} (1 STATIC OBSTACLE, MOVING GOAL) ---")

    # Create environment with 1 static obstacle and 1 moving goal
    env = GridWorldEnv(
        size=size,
        goals=[{'pos_init': (size - 1, size - 1), 'moving': True}],  # Goal is moving
        obstacles=[
            {'pos_init': (1, 1), 'moving': False}  # Only 1 static obstacle
        ] if size > 3 else []  # Skip for very small grids
    )

    # Trainer setup
    trainer = GridWorldTrainer(
        env=env,
        episodes=EPISODES,
        max_steps=MAX_STEPS_PER_EPISODE,
        moving_avg_window=MOVING_AVG_WINDOW
    )

    # Train the agent
    data = trainer.train()
    results.append(data)

    print(f"Q-table for grid size {size}:")
    for state_key in sorted(data['q_table'].keys()):
        print(f"State {state_key}: {data['q_table'][state_key]}")

    total_return = sum(data['rewards'])
    print(f"Total return over all episodes for grid size {size}: {total_return}")

    # Animate agent behavior
    animator = GridWorldAnimator(data['env'], data['q_table'])
    animator.animate()

# === Plot: Episodes vs Moving Average of Rewards ===
plt.figure(figsize=(10, 6))
for res in results:
    episodes = np.arange(len(res['moving_avg_rewards'])) + MOVING_AVG_WINDOW
    plt.plot(episodes, res['moving_avg_rewards'], label=f'Grid {res["grid_size"]}')
plt.title('Episodes vs Average Return (Moving Goal, 1 Static Obstacle)')
plt.xlabel('Episode')
plt.ylabel('Average Return')
plt.legend()
plt.grid()
plt.show()

# === Plot: Episodes vs Steps per Episode ===
plt.figure(figsize=(10, 6))
for res in results:
    episodes = np.arange(1, EPISODES + 1)
    plt.plot(episodes, res['steps'], label=f'Grid {res["grid_size"]}')
plt.title('Episodes vs Steps per Episode (Moving Goal)')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.legend()
plt.grid()
plt.show()

# === Plot: Final Performance vs Grid Size ===
final_performances = [np.mean(res['rewards'][-100:]) for res in results]

plt.figure(figsize=(8, 5))
plt.plot(grid_sizes, final_performances, marker='o')
plt.title('Final Performance vs Grid Size (Moving Goal)')
plt.xlabel('Grid Size')
plt.ylabel('Avg Return (Last 100 Episodes)')
plt.grid()
plt.show()
