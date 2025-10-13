# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

# analysis.py
# -----------

######################
# ANALYSIS QUESTIONS #
######################

def question2a():
    """
    Prefer the close exit (+1), risking the cliff (-10).
    Expected policy: agent takes risky path near cliff.
    """
    answerDiscount = 0.3      
    answerNoise = 0.0         
    answerLivingReward = -0.2
    return answerDiscount, answerNoise, answerLivingReward


def question2b():
    """
    Prefer the close exit (+1), avoiding the cliff (-10).
    Expected policy: agent takes safe path away from cliff.
    """
    answerDiscount = 0.3      
    answerNoise = 0.2        
    answerLivingReward = -0.2 
    return answerDiscount, answerNoise, answerLivingReward


def question2c():
    """
    Prefer the distant exit (+10), risking the cliff (-10).
    Agent values distant exit, and is willing to risk cliff.
    """
    answerDiscount = 0.9      
    answerNoise = 0.0        
    answerLivingReward = 0.0  
    return answerDiscount, answerNoise, answerLivingReward

def question2d():
    """
    Prefer the distant exit (+10), avoiding the cliff (-10).
    Agent values distant exit but avoids cliff due to noise.
    """
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question2e():
    """
    Avoid both exits and cliff; never terminate.
    Agent prefers living forever rather than exiting.
    """
    answerDiscount = 1.0     
    answerNoise = 0.0         
    answerLivingReward = 2.0  
    return answerDiscount, answerNoise, answerLivingReward
