# Livrable_1&2 & GW & Pacman AI Project & GridWorld-RL-Experiments : Reinforcement Learning Projects

This repository contains fou main projects focused on Reinforcement Learning (RL) agents navigating environments with increasing complexity.

---

## Livrable_1&2: GridWorld Environment V0 and RL Agents

This part includes a basic GridWorld environment (`GridWorldEnvV0.py`) and implementations of several classic RL algorithms.

### Environment V0 Details
| Feature          | Detail                       |
|------------------|------------------------------|
| Grid Size        | 5x5                          |
| Start Position   | (0, 0) - Top-left corner     |
| Goal Position    | (4, 4) - Bottom-right corner |
| Actions          | Up, Down, Left, Right        |
| Movement Reward  | -0.5 per step                |
| Goal Reward      | +50 (episode terminates)     |
| Boundary Rule    | Agent cannot move outside grid|

### RL Algorithms Implemented:
- Policy Iteration (`policy_iteration_agent.py`)
- Value Iteration (`value_iteration_agent.py`)
- Monte Carlo (`monte_carlo_agent.py`)
- Q-Learning (`q_learning_agent.py`)

These algorithms were tested mainly on this simple, static environment to understand their behavior and performance.

---

## GW: Enhanced GridWorld Environment and Q-Learning Experiments

This is the newer, more complex GridWorld environment (`GridWorldEnv.py`) used in the **GW** project. It supports multiple goals and obstacles, some of which can move dynamically, adding new challenges for RL agents.

### Key Enhancements in GW Environment
| Feature            | Detail                                                                 |
|--------------------|------------------------------------------------------------------------|
| Multiple Goals     | Supports multiple goal states with configurable initial positions       |
| Obstacles          | Multiple obstacles that can be static or moving                         |
| Moving Entities    | Both goals and obstacles can be dynamic, changing positions during episodes |
| Collision Penalty  | -10 reward for colliding with obstacles; agent stays in place          |
| Goal Reward        | +30 reward for reaching any goal, ending the episode                   |
| Step Cost          | -0.1 reward per step to encourage efficient navigation                  |
| Visualization      | Text-based grid render showing agent (A), goals (G), obstacles (X), and free spaces (.) |

### Q-Learning Experiments in GW

In this project, Q-Learning was the main RL method applied to navigate increasingly complex scenarios:
- Starting with static obstacles and goals
- Progressing to multiple goals and obstacles
- Finally tackling the challenge of **moving obstacles and goals**, which introduce stochastic dynamics and partial observability

Due to the difficulty posed by moving entities, Deep Learning approaches were explored to better approximate the Q-function and improve learning stability and performance.


# Pacman AI Project (UC Berkeley)

This project provides a framework for building AI agents that play the classic Pacman game. It includes:

- **Game Environment:** A simplified Pacman game with a grid-based maze, food pellets, walls, and ghosts.
- **Feature Extraction:** Various feature extractors to represent game states for learning agents.
- **Reflex Agents and Q-Learning:** Sample agents that make decisions based on features or learned Q-values.
- **Search and Planning:** Support for search algorithms used in pathfinding and decision making.

The goal is to implement and experiment with AI techniques like reinforcement learning, search, and feature engineering to create intelligent Pacman agents.


# GridWorld-RL-Experiments

This project builds on the same GridWorld environment developed earlier, using it as the base for new reinforcement learning experiments.  
The goal is to explore **Deep Reinforcement Learning (DRL)** methods, focusing on both **naive deep learning approaches** and the **DeepMind-style Deep Q-Learning (DQN)** implementation.

The original GridWorld structure remains unchanged.  
A separate wrapper class was added only to make it compatible with **Stable-Baselines3 (SB3)**, allowing direct comparison between custom deep RL agents and SB3 implementations such as DQN and PPO.

The experiments train agents in dynamic environments with moving goals and obstacles, evaluating how different deep learning strategies handle complexity, adaptability, and learning stability.


# SBL3-ZOO

This repository contains experiments with **Reinforcement Learning (RL)** and **Deep Reinforcement Learning (DRL)** in custom GridWorld environments, building upon previous projects like Livrable_1&2, GW, and Pacman AI.

The custom GridWorld environment is flexible and supports configurable grid sizes, multiple goals and obstacles, and dynamic/moving goals and obstacles. It also includes step costs, collision penalties, and goal rewards. A text-based visualization shows the agent (A), goals (G), and obstacles (X).

A wrapper (`make_sb3_env`) was created to make the custom GridWorld compatible with **Stable-Baselines3 (SB3)**. Agents can be trained using standard SB3 algorithms such as **PPO** (Proximal Policy Optimization), **DQN** (Deep Q-Learning), and **A2C** (Advantage Actor-Critic). Training logs can be visualized using **TensorBoard**.

Default hyperparameters from `FrozenLake-v1` were extracted for PPO and DQN. These parameters were used to initialize GridWorld DRL agents to leverage proven settings and improve training stability and performance.

Agents were trained in environments with static goals and obstacles, as well as moving goals and obstacles. The **GridWorldAnimator** class was used to animate agent behavior, showing how learned policies navigate the environment step by step. The experiments successfully trained PPO, DQN, and A2C agents in dynamic GridWorlds. Hyperparameters from classical benchmarks were reused to accelerate DRL training, and the modular structure allows swapping algorithms easily and testing new RL methods.


