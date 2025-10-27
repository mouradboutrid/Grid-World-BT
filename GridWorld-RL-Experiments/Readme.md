# GridWorld Reinforcement Learning Experiments

This repository contains a collection of **Reinforcement Learning (RL)** experiments implemented on a **custom dynamic GridWorld environment**.  
It includes implementations of **Q-learning**, **Deep Q-Learning (DQN)**, **naïve deep agents**, and **Stable-Baselines3 (SB3)** integration for benchmark comparisons.

---

## Structure

| File | Description |
|------|--------------|
| `GridWorldEnv.py` | Core environment class defining the dynamic GridWorld with moving goals and obstacles. |
| `GridWorldAnimator.py` | Utility for visualizing and animating agent movements and environment evolution. |
| `q-learning.py` | Classical tabular Q-learning implementation on the GridWorld environment. |
| `dm_deepQL.py` | Custom Deep Q-Learning implementation (manual neural network). |
| `naiveDL_agent.py` | Simple Deep Learning agent using basic network structures without optimization libraries. |
| `naiveDL_agent_SB3.py` | SB3-compatible version of the naïve Deep Learning agent. |
| `mydeepQL_SB3.py` | Deep Q-Learning setup using **Stable-Baselines3** for benchmarking and performance comparison. |
| `SB3_env.py` | Adapter for converting the original environment into SB3-compatible Gym environment. |
| `comparaison.py` | Script for comparing performance metrics across all RL methods (Q-learning, DQN, SB3). |

---

## It study:

- Deep reiforcement learning behaviere in a Dynamic **GridWorld** with moving obstacles and goals  
- Supports both **manual agents** and **Stable-Baselines3** agents  
- Modular environment compatible with Gym API    
- Performance comparison 

---
