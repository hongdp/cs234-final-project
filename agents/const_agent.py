import numpy as np
from .agent import Agent, Actions

class ConstAgent(Agent):

    def __init__(self, config, dataset):
        pass

    def act(self, feature):
        '''
        return: action, context
        '''
        return Actions.MEDIUM, None

    def feedback(self, reward, context):
        pass
