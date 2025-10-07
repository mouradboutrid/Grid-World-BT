import numpy as np
import matplotlib.pyplot as plt
from GridWorldEnv import GridWorldEnv
from GW_AG import GridWorldTrainer, GridWorldAnimator

# === Enhanced Trainer with Sophisticated Distance Reward ===
class DistanceAwareGridWorldTrainer(GridWorldTrainer):
    def __init__(self, *args, **kwargs):
        self.distance_weight = kwargs.pop('distance_weight', 0.5)
        self.progress_bonus = kwargs.pop('progress_bonus', 1.0)
        self.regression_penalty = kwargs.pop('regression_penalty', 0.5)
        super().__init__(*args, **kwargs)

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _euclidean_distance(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _get_closest_goal_distance(self, agent_pos):
        """Get distance to closest goal"""
        goal_positions = self.env.get_goal_positions()
        if not goal_positions:
            return 0
        return min(self._manhattan_distance(agent_pos, goal) for goal in goal_positions)

    def _calculate_distance_reward(self, prev_pos, current_pos, goal_pos):
        """Calculate sophisticated distance-based reward"""
        # Multiple distance metrics
        old_manhattan = self._manhattan_distance(prev_pos, goal_pos)
        new_manhattan = self._manhattan_distance(current_pos, goal_pos)
        
        old_euclidean = self._euclidean_distance(prev_pos, goal_pos)
        new_euclidean = self._euclidean_distance(current_pos, goal_pos)
        
        # Progress metrics
        manhattan_progress = old_manhattan - new_manhattan
        euclidean_progress = old_euclidean - new_euclidean
        
        # Combined progress (weighted)
        progress = (manhattan_progress * 0.7 + euclidean_progress * 0.3)
        
        # Distance-based reward components
        distance_reward = 0.0
        
        # Progress bonus/penalty
        if progress > 0:
            distance_reward += self.progress_bonus * progress
        elif progress < 0:
            distance_reward += self.regression_penalty * progress
        
        # Absolute distance penalty (encourage getting closer in general)
        max_possible_distance = 2 * self.env.size  # Approximate max Manhattan distance
        normalized_distance = new_manhattan / max_possible_distance
        distance_reward -= self.distance_weight * normalized_distance
        
        # Bonus for significant progress
        if manhattan_progress >= 2:  # Moved significantly closer
            distance_reward += 0.5
        
        return distance_reward

    def train(self):
        for episode in range(1, self.episodes + 1):
            state = self.env.reset()
            state_key = self.agent.get_state_key(state)
            total_reward = 0
            episode_progress = 0

            # Get goal position (assuming single goal for simplicity)
            goal_positions = self.env.get_goal_positions()
            if not goal_positions:
                goal_pos = (self.env.size - 1, self.env.size - 1)  # Default goal
            else:
                goal_pos = goal_positions[0]

            for step in range(self.max_steps):
                prev_agent_pos = tuple(self.env.agent_pos)

                action = self.agent.choose_action(state_key)
                next_state, environment_reward, done, _ = self.env.step(action)
                next_state_key = self.agent.get_state_key(next_state)
                current_agent_pos = tuple(self.env.agent_pos)

                # Calculate distance-based reward
                distance_reward = self._calculate_distance_reward(
                    prev_agent_pos, current_agent_pos, goal_pos
                )
                
                # Combine environment reward with distance reward
                total_step_reward = environment_reward + distance_reward
                
                # Additional bonuses for good behavior
                if environment_reward == 30:  # Reached goal
                    total_step_reward += 20  # Extra success bonus
                
                # Update Q-table with combined reward
                self.agent.update_q_table(
                    state_key, action, total_step_reward, next_state_key, done
                )

                state_key = next_state_key
                total_reward += total_step_reward
                episode_progress += distance_reward

                if done:
                    # Additional completion bonus
                    completion_bonus = max(0, 50 - step)  # Bonus for faster completion
                    total_reward += completion_bonus
                    break

            self.rewards_all_episodes.append(total_reward)
            self.steps_all_episodes.append(step + 1)
            self.agent.decay_epsilon()
            
            # Log progress every 100 episodes
            if episode % 100 == 0:
                avg_progress = episode_progress / min(step + 1, self.max_steps)
                print(f"Episode {episode}: Total Reward = {total_reward:.2f}, "
                      f"Avg Progress = {avg_progress:.3f}")

        # Moving average for plotting
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
            'grid_size': self.grid_size,
            'distance_weight': self.distance_weight
        }

# === Hyperparameters ===
EPISODES = 2000
MAX_STEPS_PER_EPISODE = 100
MOVING_AVG_WINDOW = 50

# === Grid sizes to evaluate ===
grid_sizes = [4, 6, 8, 10]
results = []

# Different distance reward configurations to test
configs = [
    {'distance_weight': 0.3, 'progress_bonus': 1.0, 'regression_penalty': 0.3},
    {'distance_weight': 0.5, 'progress_bonus': 1.5, 'regression_penalty': 0.5},
    {'distance_weight': 0.7, 'progress_bonus': 2.0, 'regression_penalty': 0.7}
]

for config_idx, config in enumerate(configs):
    print(f"\n=== Testing Configuration {config_idx + 1} ===")
    print(f"Distance Weight: {config['distance_weight']}, "
          f"Progress Bonus: {config['progress_bonus']}, "
          f"Regression Penalty: {config['regression_penalty']}")
    
    config_results = []
    
    for size in grid_sizes:
        print(f"\n--- Training for grid size {size} ---")

        env = GridWorldEnv(
            size=size,
            goals=[{'pos_init': (size - 1, size - 1), 'moving': True}],
            obstacles=[{'pos_init': (1, 1), 'moving': False}] if size > 3 else []
        )

        trainer = DistanceAwareGridWorldTrainer(
            env=env,
            episodes=EPISODES,
            max_steps=MAX_STEPS_PER_EPISODE,
            moving_avg_window=MOVING_AVG_WINDOW,
            **config
        )

        data = trainer.train()
        data['config'] = config
        config_results.append(data)

        print(f"Completed grid size {size}")
        final_avg_reward = np.mean(data['rewards'][-100:])
        print(f"Final average reward (last 100 episodes): {final_avg_reward:.2f}")

    results.append(config_results)

# === Enhanced Plotting ===

# Plot: Compare configurations for each grid size
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for idx, size in enumerate(grid_sizes):
    ax = axes[idx]
    for config_idx, config_results in enumerate(results):
        if idx < len(config_results):
            data = config_results[idx]
            episodes = np.arange(len(data['moving_avg_rewards'])) + MOVING_AVG_WINDOW
            label = f"Config {config_idx+1} (w={data['distance_weight']})"
            ax.plot(episodes, data['moving_avg_rewards'], label=label, linewidth=2)
    
    ax.set_title(f'Grid Size {size} - Learning Curves')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Return')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot: Final performance comparison
plt.figure(figsize=(12, 6))

# Subplot 1: Performance vs Grid Size for each configuration
plt.subplot(1, 2, 1)
for config_idx, config_results in enumerate(results):
    final_performances = [np.mean(res['rewards'][-100:]) for res in config_results]
    config_label = f"Config {config_idx+1}"
    plt.plot(grid_sizes, final_performances, marker='o', linewidth=2, label=config_label)

plt.title('Final Performance vs Grid Size')
plt.xlabel('Grid Size')
plt.ylabel('Avg Return (Last 100 Episodes)')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Steps to completion
plt.subplot(1, 2, 2)
for config_idx, config_results in enumerate(results):
    avg_steps = [np.mean(res['steps'][-100:]) for res in config_results]
    config_label = f"Config {config_idx+1}"
    plt.plot(grid_sizes, avg_steps, marker='s', linewidth=2, label=config_label)

plt.title('Average Steps vs Grid Size')
plt.xlabel('Grid Size')
plt.ylabel('Steps per Episode')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# === Animate best performing configuration ===
best_overall_reward = -np.inf
best_config_data = None

for config_results in results:
    for data in config_results:
        avg_reward = np.mean(data['rewards'][-100:])
        if avg_reward > best_overall_reward:
            best_overall_reward = avg_reward
            best_config_data = data

if best_config_data:
    print(f"\n=== Animating Best Configuration ===")
    print(f"Grid Size: {best_config_data['grid_size']}")
    print(f"Average Reward: {best_overall_reward:.2f}")
    
    animator = GridWorldAnimator(best_config_data['env'], best_config_data['q_table'])
    animator.animate()