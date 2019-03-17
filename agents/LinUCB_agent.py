import numpy as np
from .agent import Agent, Actions


class LinUCBAgent(Agent):
    def __init__(self, config, vocab):
        self.vocab = vocab
        # self.history = []
        # self.total = 0
        self.A = np.stack([np.identity(29) for _ in range(3)], axis=0)
        self.b = np.zeros((3,29))

        self.alpha = 2.15
        #pass

    def act(self, feature):
        p = np.zeros(3)
        for ind in range(3):
            invA = np.linalg.inv(self.A[ind])
            theta = np.matmul(invA, self.b[ind])
            p[ind] = np.matmul(feature, theta) + self.alpha * np.sqrt(np.matmul(np.matmul(invA, feature), feature))

        action = np.argmax(p)

        if action == 0:
            return Actions.LOW
        elif action == 1:
            return Actions.MEDIUM
        else:
            return Actions.HIGH
        #pass

    def feedback(self, feature, reward, action):
        # print(reward, action.value)
        self.A[action.value] += np.dot(feature, feature)
        self.b[action.value] += reward * feature
        #pass