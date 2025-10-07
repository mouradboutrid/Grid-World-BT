# Livrable_1&2: GridWorld Reinforcement Learning Agents

This project focuses on implementing and comparing several fundamental **Reinforcement Learning (RL) algorithms** to solve a navigation task within a **GridWorld** environment. The goal is for an agent to find the optimal path from the start to the goal state.

## 🗺️ GridWorld Environment

The environment, primarily defined in `GridWorldEnvV0.py`, is a basic Markov Decision Process (MDP) designed for RL experimentation. A second version of the environment is implied, likely containing minor adjustments for comparative study.

### Environment Details

| Feature | Detail |
| :--- | :--- |
| **Grid Size** | 5x5 |
| **Start Position** | (0, 0) - Top-left corner |
| **Goal Position** | (4, 4) - Bottom-right corner |
| **Actions** | Up, Down, Left, Right |
| **Movement Reward** | **-0.5** per step (small penalty to encourage efficiency) |
| **Goal Reward** | **+50** (terminating reward) |
| **Boundary Rule** | Agent cannot move outside the grid boundaries. |

The environment is implemented in **Python** and includes visualization capabilities using **Matplotlib** to show the grid, agent's position, and the goal state.

***

## 🧠 Reinforcement Learning Algorithms

Various RL techniques, categorized as Dynamic Programming (Model-Based) and Model-Free methods, were implemented to solve the GridWorld.

### Implemented Algorithms

| File | Algorithm | Type | Key Characteristic |
| :--- | :--- | :--- | :--- |
| `policy_iteration_agent.py` | **Policy Iteration** | Model-Based | Alternates between policy evaluation and policy improvement. **Guaranteed to converge** to optimal policy. |
| `value_iteration_agent.py` | **Value Iteration** | Model-Based | Iteratively updates the value function using the **Bellman Optimality Equation**. Efficient when transition dynamics are known. |
| `monte_carlo_agent.py` | **Monte Carlo (MC)** | Model-Free | Learns from **complete episodes** (returns). Uses experience sampling to estimate value functions. |
| `q_learning_agent.py` | **Q-Learning** | Model-Free (TD) | **Temporal Difference** learning. Learns the optimal action-value function ($Q^*$) directly, independent of the current policy (off-policy). |

***

