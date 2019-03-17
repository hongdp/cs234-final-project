import numpy as np
from .agent import Agent, Actions


class LinUCBAgent(Agent):
    def __init__(self, config, vocab):
        self.vocab = vocab
        self.history = []
        self.total = 0
        pass

    def act(self, feature):
        pass

    def feedback(self, feature, reward):

        pass