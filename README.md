# Livrable_1&2 & GW: GridWorld Reinforcement Learning Projects

This repository contains two main projects focused on Reinforcement Learning (RL) agents navigating GridWorld environments of increasing complexity.

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

## 📁 Project Structure

