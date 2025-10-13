# Pacman AI Project (UC Berkeley)

This project provides a framework for building AI agents that play the classic Pacman game. It includes:

- **Game Environment:** A simplified Pacman game with a grid-based maze, food pellets, walls, and ghosts.
- **Feature Extraction:** Various feature extractors to represent game states for learning agents.
- **Reflex Agents and Q-Learning:** Sample agents that make decisions based on features or learned Q-values.
- **Search and Planning:** Support for search algorithms used in pathfinding and decision making.

My goal is to implement and experiment with AI techniques like reinforcement learning, search, and feature engineering to create intelligent Pacman agents.

---

## My Enhanced Feature Extractor

I extended the original feature extractor (`SimpleExtractor`) with an `EnhancedExtractor` that adds new features to improve Pacman’s decision-making:

### Features I Added

- **Ghost Awareness 2 Steps Away:** I count ghosts within 2 steps to help Pacman avoid dangers earlier.
- **Food Clusters Within 2 Steps:** I measure how many food pellets are close by, encouraging Pacman to move toward clusters of food.
- **Movement Towards Food:** I reward actions that bring Pacman closer to the nearest food pellet, promoting purposeful movement.

---

## What I Expect to See

By using the `EnhancedExtractor` with an approximate Q-learning agent, I expect:

- **Better Food Collection:** Pacman targets clusters of food and moves more strategically instead of focusing only on immediate pellets.
- **Smarter Ghost Avoidance:** Pacman anticipates ghost positions two steps ahead, resulting in safer paths and fewer deaths.
- **Longer Survival:** Avoiding ghosts earlier and better food targeting helps Pacman survive longer in tougher mazes.
- **More Efficient Navigation:** Pacman moves purposefully, reducing random wandering and improving overall performance.

---

## How I Run It

To run the Pacman agent with my enhanced features, I use:

```bash
python pacman.py -p ApproximateQAgent -a extractor=EnhancedExtractor -n 20 -l mediumGrid
```

And to compare with the original feature extractor:
```bash
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -n 20 -l mediumGrid
