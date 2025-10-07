# GridWorld Homework - Approach Descriptions

---

## Test 1: Static Obstacles with Q-Learning

**Approach:**  
The agent was trained in environments with static obstacles and a fixed goal location across various grid sizes (4x4, 6x6, 8x8, 10x10). The obstacles remain stationary, allowing the agent to learn reliable paths around them.

**What’s new:**  
- Baseline use of Q-learning in static environments with multiple fixed obstacles.  
- Evaluation of agent performance as grid size increases.

**Purpose:**  
To validate the Q-learning agent’s ability to solve static GridWorld environments and establish a baseline for future experiments.

**Results:**  
- Agent successfully learned to navigate to the goal while avoiding obstacles.  
- Q-tables and learning curves showed steady improvement over episodes.  
- Animated simulations confirmed effective navigation policies.

---

## Test 2: Moving Obstacles with Q-Learning

**Approach:**  
This test introduced moving obstacles while keeping the goal static. Obstacles change positions dynamically at each step, adding stochasticity to the environment.

**What’s new:**  
- Agent faced the challenge of navigating in a non-stationary environment.  
- Tested whether Q-learning can adapt to dynamic changes in obstacles.

**Purpose:**  
To assess robustness of the Q-learning policy when obstacles are not fixed and require dynamic avoidance strategies.

**Results:**  
- Agent showed learning but with slower convergence compared to static obstacle scenario.  
- Q-table updates reflected adjustments to the moving obstacles.  
- Visualizations showed the agent learning to avoid dynamic obstacles effectively.

---

## Test 3: Moving Goal with Static Obstacles (No Goal Position Info, No Effective Learning)

**Approach:**  
The agent was placed in an environment with one static obstacle and a **moving goal**, but crucially **without providing the agent with any direct information about the goal’s position**.

**What’s new:**  
- Introduced a moving goal with unknown location to the agent, increasing the task complexity significantly.

**Purpose:**  
To test the agent’s ability to learn in a scenario where the goal moves unpredictably and the agent has no prior knowledge of its current location.

**Results:**  
- The agent failed to learn meaningful policies.  
- Rewards and Q-values showed no significant improvement.  

---

## Test 4: Distance-Aware Reward Shaping with Moving Goal (With Goal Position Info)

**Approach:**  
An enhanced training method incorporated a distance-based reward signal combining Manhattan and Euclidean distances to guide the agent towards the moving goal. However, in this setup, the agent **was given information about the current goal position**, effectively “cheating” by relying on this explicit data for the reward calculations, bypassing the need for the agent to learn or explore the environment.

**What’s new:**  
- Reward shaping was used to provide immediate, incremental feedback, offering more detailed guidance than the sparse terminal rewards.  
- The agent exploited the goal position info to compute distance-based progress rewards, essentially manipulating the reward structure to maximize efficiency without learning generalizable strategies.  
- Multiple configurations of reward weights and penalties were tested to fine-tune the reward shaping, but the overall goal was to artificially enhance the agent's performance without encouraging actual learning.

**Purpose:**  
The aim was not to promote true learning but to artificially guide the agent's behavior by giving it explicit goal information, allowing it to "cheat" its way through the task by focusing on short-term progress towards the goal.

**Results:**  
- The agent showed some **apparent improvement** in performance, but this was more of an exploitation of the provided goal information than genuine learning.  
- Training curves were better than naive methods without reward shaping, but they still represented **mechanical exploitation of reward signals**, rather than the agent truly learning how to achieve the goal.  
- Despite the addition of goal position information, performance remained **unstable and limited**, suggesting that the agent’s reliance on explicit goal knowledge prevented the development of more robust, generalizable strategies.


---
