import numpy as np
from .agent import Agent

class ConstAgent(Agent):

    def __init__(self, config, vocab):
        pass

    def act(self, feature):
        return 35.0

    def feedback(self, feature, reward):
        pass
