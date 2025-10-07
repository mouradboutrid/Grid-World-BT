# Approach Descriptions

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

## Test 3: Moving Goal with Static Obstacles (No Effective Learning)

**Approach:**  
The agent was placed in an environment with one static obstacle but a **moving goal** that changes position dynamically every episode or step.

**What’s new:**  
- Introduced a goal that moves, requiring the agent to continuously track and adapt to changing target locations.

**Purpose:**  
To challenge the agent’s ability to track and reach a moving goal instead of a fixed one.

**Results:**  
- The agent failed to learn meaningful policies in this setup.  
- Rewards and Q-values showed no significant improvement over time.  
- The moving goal’s unpredictability appeared too complex for standard Q-learning without additional strategies.

---

## Test 4: Distance-Aware Reward Shaping with Moving Goal (No Effective Learning)

**Approach:**  
An enhanced training method that incorporated a sophisticated distance-based reward signal (combining Manhattan and Euclidean distances) to better guide the agent towards the moving goal.

**What’s new:**  
- Reward shaping aimed to provide incremental feedback to improve learning.  
- Tested multiple configurations to tune reward weights and penalties.

**Purpose:**  
To improve learning in moving goal scenarios by offering more informative rewards rather than sparse terminal feedback.

**Results:**  
- Despite the advanced reward shaping, the agent still did not learn effective policies.  
- Moving goal complexity remained a major challenge.  
- Training curves showed limited improvement; agent struggled to converge.

---
