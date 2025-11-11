# SBL3-ZOO

## Experiments Overview

Before moving to the custom environment, multiple classical games were used to understand RL behavior and train agents:

| Environment       | Algorithm(s)        | Purpose / Observations                                                                 |
|------------------|-------------------|----------------------------------------------------------------------------------------|
| MountainCar-v0    | PPO, A2C           | Tested policy-based algorithms; observed learning to climb hill using momentum.       |
| Breakout          | PPO, DQN           | Tested value-based vs policy-based agents; agent learned to control paddle effectively.|
| FrozenLake-v1     | PPO, DQN           | Extracted default hyperparameters for later use; understood exploration/exploitation trade-off. |

**Key Lessons from Initial Experiments:**
- Policy-based (PPO, A2C) and value-based (DQN) algorithms have different learning dynamics.  
- TensorBoard was used to monitor rewards, losses, and learning progress.  

---

## Custom GridWorld Environment

After gaining familiarity with RL concepts in standard Gym environments, i try with my own **custom GridWorld environment** designed to be flexible and challenging:

**Environment Features:**
-as the previous repos projects 

**Approaches Applied:**
1. **Classical Q-Learning** – Tested in static and simple scenarios to validate reward shaping and navigation.  
2. **Deep RL (SB3)** – Used a wrapper to make the environment compatible with Stable-Baselines3. Algorithms applied:
   - **PPO (Proximal Policy Optimization)**  
   - **DQN (Deep Q-Learning)**  
   - **A2C (Advantage Actor-Critic)**  
3. **Hyperparameter Initialization** – Leveraged extracted FrozenLake hyperparameters to accelerate convergence and stabilize learning.

---

## Training and Evaluation

Agents were trained in **phases**, gradually increasing environment complexity:

| Training Phase             | Environment Configuration          | Algorithms Applied | Observations / Goals |
|----------------------------|----------------------------------|-----------------|--------------------|
| Phase 1: Static Environment | Single goal, static obstacles    | PPO, DQN, A2C  | Agent learned basic navigation and goal-reaching behavior. |
| Phase 2: Multiple Goals/Obstacles | Two goals, multiple static obstacles | PPO, DQN, A2C | Tested agent’s ability to choose optimal path and avoid collisions. |
| Phase 3: Moving Entities  | Moving goals and moving obstacles | PPO, DQN, A2C  | Evaluated adaptability; agents needed to handle stochastic dynamics and partial observability. |

---

## Achievements

- Successfully trained **PPO, DQN, and A2C agents** in both static and dynamic GridWorld environments.  
- Demonstrated the effect **hyperparameters from classical benchmarks** to costum environments.  
- vesualise the result and training progress using TensorBoard


---


