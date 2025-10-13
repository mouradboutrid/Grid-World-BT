# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import random
import util


class QLearningAgent(ReinforcementAgent):
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.qValues = util.Counter()  # dictionary with default 0 for Q-values

    def getQValue(self, state, action):
        """
        Returns Q(state,action)
        Should return 0.0 if we have never seen a state or the Q node value otherwise
        """
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions. If no legal actions (terminal), return 0.0
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0
        return max(self.getQValue(state, action) for action in legalActions)

    def computeActionFromQValues(self, state):
        """
        Compute the best action to take in a state.
        If no legal actions (terminal), return None.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        maxValue = self.computeValueFromQValues(state)
        bestActions = [action for action in legalActions if self.getQValue(state, action) == maxValue]
        return random.choice(bestActions)

    def getAction(self, state):
        """
        Compute the action to take in the current state.
        With probability self.epsilon, take a random action,
        else take the best policy action.
        If no legal actions, return None.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
        Called to observe a state = action => nextState and reward transition.
        Do your Q-Value update here.
        """
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.qValues[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample


class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as QLearningAgent, but with different default parameters.
    """

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the parent getAction, then informs parent of action for Pacman-specific bookkeeping.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        PacmanQAgent.__init__(self, **args)
        self.featExtractor = util.lookup(extractor, globals())()
        self.weights = util.Counter()

    def getWeights(self):
        """
        Return weights dictionary for autograder and debugging.
        """
        return self.weights

    def getQValue(self, state, action):
        """
        Return Q(state,action) = w * featureVector
        """
        features = self.featExtractor.getFeatures(state, action)
        q_value = 0.0
        for feature, value in features.items():
            q_value += self.weights[feature] * value
        return q_value

    def update(self, state, action, nextState, reward):
        """
        Update weights based on transition.
        """
        features = self.featExtractor.getFeatures(state, action)
        legalActions = self.getLegalActions(nextState)
        nextValue = 0.0
        if legalActions:
            nextValue = max(self.getQValue(nextState, next_action) for next_action in legalActions)
        difference = (reward + self.discount * nextValue) - self.getQValue(state, action)

        for feature, value in features.items():
            self.weights[feature] += self.alpha * difference * value

    def final(self, state):
        """
        Called at the end of each game.
        You can print weights for debugging after training is done.
        """
        if self.episodesSoFar == self.numTraining:
            print("Final weights:", self.weights)
        PacmanQAgent.final(self, state)


__all__ = ['PacmanQAgent', 'ApproximateQAgent']
