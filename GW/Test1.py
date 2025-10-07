import numpy as np
import matplotlib.pyplot as plt
from GridWorldEnv import GridWorldEnv
from GW_AG import GridWorldTrainer, GridWorldAnimator 

# === Hyperparameters ===
EPISODES = 2000
MAX_STEPS_PER_EPISODE = 100
MOVING_AVG_WINDOW = 50

# === Grid sizes to evaluate ===
grid_sizes = [4, 6, 8, 10]
results = []

for size in grid_sizes:
    print(f"\n--- Training for grid size {size} (STATIC, 3 Obstacles) ---")

    # Create a static environment with 3 obstacles
    static_env = GridWorldEnv(
        size=size,
        goals=[{'pos_init': (size - 2, size - 1), 'moving': False}],
        obstacles=[
            {'pos_init': (1, 1), 'moving': False},
            {'pos_init': (2, 2), 'moving': False},
            {'pos_init': (1, 3), 'moving': False}
        ] if size > 3 else []  # Avoid placing obstacles in tiny grids
    )

    # Create trainer with this environment
    trainer = GridWorldTrainer(
        env=static_env,
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
plt.title('Episodes vs Average Return (Moving Average)')
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
plt.title('Episodes vs Steps per Episode')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.legend()
plt.grid()
plt.show()

# === Plot: Final Performance vs Grid Size ===
final_performances = [np.mean(res['rewards'][-100:]) for res in results]

plt.figure(figsize=(8, 5))
plt.plot(grid_sizes, final_performances, marker='o')
plt.title('Final Performance vs Grid Size')
plt.xlabel('Grid Size')
plt.ylabel('Avg Return (Last 100 Episodes)')
plt.grid()
plt.show()