import numpy as np
from .agent import Agent, Actions

class ConstAgent(Agent):

    def __init__(self, config, vocab):
        pass

    def act(self, feature):
        return Actions.MEDIUM

    def feedback(self, feature, reward):
        pass
