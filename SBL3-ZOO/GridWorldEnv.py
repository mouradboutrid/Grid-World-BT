import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

class GridWorldEnv:
    def __init__(self, size=5, goals=None, obstacles=None):
        self.size = size
        self.action_space = ['up', 'down', 'left', 'right']

        # Initialize goals
        self.goals = []
        if goals:
            for g in goals:
                self.goals.append({
                    'pos_init': g['pos_init'],
                    'pos': g['pos_init'],
                    'moving': g.get('moving', False)
                })

        # Initialize obstacles
        self.obstacles = []
        if obstacles:
            for o in obstacles:
                self.obstacles.append({
                    'pos_init': o['pos_init'],
                    'pos': o['pos_init'],
                    'moving': o.get('moving', False)
                })

        self.episode_count = 0  # Track episodes
        self.reset()

    def seed(self, seed_val):
        random.seed(seed_val)

    def reset(self):
        self.agent_pos = (0, 0)
        self.done = False
        self.episode_count += 1

        # Reset or move obstacles
        for o in self.obstacles:
            if o['moving']:
                o['pos'] = self._find_free_cell({self.agent_pos} | {g['pos'] for g in self.goals})
            else:
                o['pos'] = o['pos_init']

        # Reset or move goals
        for g in self.goals:
            if g['moving']:
                g['pos'] = self._find_free_cell({self.agent_pos} | {o['pos'] for o in self.obstacles})
            else:
                g['pos'] = g['pos_init']

        # Ensure no overlaps between all entities
        self._resolve_conflicts()

        return self.agent_pos

    def _resolve_conflicts(self):
        occupied = {self.agent_pos}

        # Adjust goal positions if overlapping
        for g in self.goals:
            if g['pos'] in occupied:
                g['pos'] = self._find_free_cell(occupied)
            occupied.add(g['pos'])

        # Adjust obstacle positions if overlapping
        for o in self.obstacles:
            if o['pos'] in occupied:
                o['pos'] = self._find_free_cell(occupied)
            occupied.add(o['pos'])

    def _find_free_cell(self, occupied):
        free = [(x, y) for x in range(self.size) for y in range(self.size)
                if (x, y) not in occupied]
        if not free:
            raise Exception("No free cells left!")
        return random.choice(free)

    def valid_position(self, pos, exclude_agent=True):
        x, y = pos
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        if exclude_agent and pos == self.agent_pos:
            return False
        if any(g['pos'] == pos for g in self.goals):
            return False
        if any(o['pos'] == pos for o in self.obstacles):
            return False
        return True

    def step(self, action):
        """
        Take a step in the environment.

        Returns:
            new_pos (tuple): New agent position
            reward (int): Reward from the action
            done (bool): Whether the episode is over
            info (dict): Additional info (currently empty)
        """
        if self.done:
            return self.agent_pos, 0, True, {}

        # Compute new position
        x, y = self.agent_pos
        if action == 'up':
            x = max(0, x - 1)
        elif action == 'down':
            x = min(self.size - 1, x + 1)
        elif action == 'left':
            y = max(0, y - 1)
        elif action == 'right':
            y = min(self.size - 1, y + 1)

        new_pos = (x, y)

        # Handle collisions and rewards
        if any(o['pos'] == new_pos for o in self.obstacles):
            reward = -10  # Hit obstacle
            new_pos = self.agent_pos
        elif any(g['pos'] == new_pos for g in self.goals):
            reward = 30  # Reached goal
            self.done = True
        else:
            reward = -0.1  # Step cost

        self.agent_pos = new_pos
        return self.agent_pos, reward, self.done, {}

    def get_goal_positions(self):
        return [g['pos'] for g in self.goals]

    def get_obstacle_positions(self):
        return [o['pos'] for o in self.obstacles]

    def render(self):
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]

        for o in self.obstacles:
            x, y = o['pos']
            grid[x][y] = 'X'

        for g in self.goals:
            x, y = g['pos']
            grid[x][y] = 'G'

        ax, ay = self.agent_pos
        grid[ax][ay] = 'A'

        print("\nGrid:")
        for row in grid:
            print(" ".join(row))
        print()
